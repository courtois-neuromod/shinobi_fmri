from shinobi_fmri.config import DATA_PATH, FIG_PATH, CONDITIONS, LOW_LEVEL_CONDITIONS, SUBJECTS as CONFIG_SUBJECTS
import argparse
import os
import os.path as op
import time
import pickle
import gc
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sbn
from nilearn import image
from nilearn.maskers import NiftiMasker
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from shinobi_fmri.utils.logger import AnalysisLogger
from shinobi_fmri.utils.provenance import create_metadata
import logging
import json
import re

# Default contrasts (game conditions) - can be overridden by --low-level-confs flag
CONTRASTS = CONDITIONS  # From config: ['HIT', 'JUMP', 'DOWN', 'LEFT', 'RIGHT', 'UP', 'Kill', 'HealthLoss']
SUBJECTS = CONFIG_SUBJECTS  # From config: ['sub-01', 'sub-02', 'sub-04', 'sub-06']
MODEL = "simple"
RESULTS_PATH = op.join(DATA_PATH, 'processed/beta_maps_correlations.pkl')


def parse_args():
    parser = argparse.ArgumentParser(description="Compute cross-dataset correlation matrix.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Number of map indices handled by this job (default: None = all).")
    parser.add_argument("--chunk-start", type=int, default=0, help="Explicit map index to start chunk from.")
    parser.add_argument("--n-jobs", type=int, default=40, help="Parallel workers for correlation computation.")
    parser.add_argument("--backend", choices=["threading", "loky"], default="loky", help="Joblib backend.")
    parser.add_argument("--log-dir", default=None, help="Directory for log files")
    parser.add_argument("--slurm", action="store_true", help="Submit SLURM jobs in chunks instead of running locally")
    parser.add_argument("--low-level-confs", action="store_true", help="Use beta maps from GLM with low-level confounds (processed_low-level/ directory)")
    return parser.parse_args()


def get_reference_geometry(path_to_data, subject):
    fname = op.join(path_to_data, "shinobi.fmriprep", subject, "ses-005", "func", f"{subject}_ses-005_task-shinobi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    img = nib.load(fname)
    aff_orig = img.affine[:, -1]
    target_affine = np.column_stack([np.eye(4, 3) * 4, aff_orig])
    return target_affine, img.shape[:3]


def build_masker(subjects, path_to_data, target_affine, target_shape):
    masks = []
    for sub in subjects:
        path = op.join(path_to_data, "cneuromod.processed", "smriprep", sub, "anat", f"{sub}_space-MNI152NLin6Asym_desc-brain_mask.nii.gz")
        masks.append(image.resample_img(
            path,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        ))
    masker = NiftiMasker()
    return masker.fit(masks)


def find_processed_sources(base_dir):
    if not op.isdir(base_dir):
        return []
    return [d for d in sorted(os.listdir(base_dir)) if op.isdir(op.join(base_dir, d))]


def make_processed_record(base_dir, source, contrast, fname, path_to_data):
    subj, ses = parse_processed_meta(fname)
    map_path = op.join(base_dir, source, contrast, fname)
    raw_path = build_raw_path(path_to_data, subj, ses)
    return {'map_path': map_path, 'raw_path': raw_path, 'subj': subj, 'ses': ses, 'cond': contrast, 'source': source}


def collect_processed_records(base_dir, contrasts, model, path_to_data, use_low_level_confs=False):
    """
    Collect beta maps from the processed directory structure.
    Includes: processed/session-level*/sub-XX/ses-YY/beta_maps/

    Args:
        base_dir: Base directory path (not used, kept for compatibility)
        contrasts: List of contrasts to collect
        model: Model name (not used, kept for compatibility)
        path_to_data: Root path to data directory
        use_low_level_confs: If True, look in processed_low-level/ instead of processed/
    """
    records = []

    # Determine which processed directory to use
    processed_dirname = "processed_low-level" if use_low_level_confs else "processed"
    processed_dir = op.join(path_to_data, processed_dirname)
    if not op.isdir(processed_dir):
        return records

    session_level_variants = [d for d in sorted(os.listdir(processed_dir))
                             if op.isdir(op.join(processed_dir, d)) and d.startswith("session-level")]

    # Iterate through each session-level variant
    for variant in session_level_variants:
        session_level_dir = op.join(processed_dir, variant)

        # Iterate through subjects
        for sub_dir in sorted(os.listdir(session_level_dir)):
            sub_path = op.join(session_level_dir, sub_dir)
            if not op.isdir(sub_path) or not sub_dir.startswith("sub-"):
                continue

            # Iterate through sessions
            for ses_dir in sorted(os.listdir(sub_path)):
                ses_path = op.join(sub_path, ses_dir)
                if not op.isdir(ses_path) or not ses_dir.startswith("ses-"):
                    continue

                # Look for beta_maps directory
                beta_maps_dir = op.join(ses_path, "beta_maps")
                if not op.isdir(beta_maps_dir):
                    continue

                # Load beta maps
                for fname in sorted(os.listdir(beta_maps_dir)):
                    if not fname.endswith("stat-beta.nii.gz"):
                        continue

                    # Parse contrast from filename
                    # Format: sub-XX_ses-YY_task-shinobi_contrast-CONDITION_stat-beta.nii.gz
                    # Use regex to handle conditions with underscores (e.g., audio_envelope)
                    try:
                        match = re.search(r'contrast-(.+?)_stat-beta\.nii\.gz', fname)
                        if not match:
                            continue
                        contrast = match.group(1)

                        if contrast not in contrasts:
                            continue

                        # Parse subject and session
                        subj = sub_dir
                        ses = ses_dir

                        map_path = op.join(beta_maps_dir, fname)
                        raw_path = build_raw_path(path_to_data, subj, ses)

                        records.append({
                            'map_path': map_path,
                            'raw_path': raw_path,
                            'subj': subj,
                            'ses': ses,
                            'cond': contrast,
                            'source': variant  # Track which session-level variant this came from
                        })
                    except Exception as e:
                        continue

    return records


def collect_subject_level_records(contrasts, path_to_data, use_low_level_confs=False):
    """
    Collect subject-level beta maps (aggregated across sessions).
    Structure: processed/subject-level/sub-XX/beta_maps/ or processed_low-level/subject-level/sub-XX/beta_maps/

    Args:
        contrasts: List of contrasts to collect
        path_to_data: Root path to data directory
        use_low_level_confs: If True, look in processed_low-level/ instead of processed/
    """
    records = []

    processed_dirname = "processed_low-level" if use_low_level_confs else "processed"
    subject_level_dir = op.join(path_to_data, processed_dirname, "subject-level")
    if not op.isdir(subject_level_dir):
        return records

    # Iterate through subjects
    for sub_dir in sorted(os.listdir(subject_level_dir)):
        sub_path = op.join(subject_level_dir, sub_dir)
        if not op.isdir(sub_path) or not sub_dir.startswith("sub-"):
            continue

        # Look for beta_maps directory
        beta_maps_dir = op.join(sub_path, "beta_maps")
        if not op.isdir(beta_maps_dir):
            continue

        # Load beta maps
        for fname in sorted(os.listdir(beta_maps_dir)):
            if not fname.endswith("stat-beta.nii.gz"):
                continue

            # Parse contrast from filename
            # Format: sub-XX_task-shinobi_contrast-CONDITION_stat-beta.nii.gz
            # Use regex to handle conditions with underscores (e.g., audio_envelope)
            try:
                match = re.search(r'contrast-(.+?)_stat-beta\.nii\.gz', fname)
                if not match:
                    continue
                contrast = match.group(1)

                if contrast not in contrasts:
                    continue

                subj = sub_dir
                map_path = op.join(beta_maps_dir, fname)

                records.append({
                    'map_path': map_path,
                    'raw_path': map_path,  # No specific raw path for subject-level
                    'subj': subj,
                    'ses': 'subject-level',  # Aggregate across all sessions
                    'cond': contrast,
                    'source': 'subject-level'
                })
            except Exception as e:
                continue

    return records


def parse_processed_meta(fname):
    parts = fname.split("_")
    subj = next((p for p in parts if p.startswith("sub-")), "")
    ses = next((p for p in parts if p.startswith("ses-")), "")
    return subj, ses


def build_raw_path(path_to_data, subj, ses):
    return op.join(path_to_data, "shinobi.fmriprep", subj, ses, "func", f"{subj}_{ses}_task-shinobi_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")


def log(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)


def summarize_sources(records):
    counts = {}
    for record in records:
        counts[record['source']] = counts.get(record['source'], 0) + 1
    return ", ".join(f"{src}: {counts[src]}" for src in sorted(counts)) if counts else "no sources"


def print_detailed_breakdown(records, logger=None):
    """Print a detailed breakdown of collected maps by category and subject."""
    from collections import defaultdict

    # Organize by source and subject
    by_source = defaultdict(lambda: defaultdict(list))
    by_subject = defaultdict(int)
    by_condition = defaultdict(int)

    for record in records:
        source = record['source']
        subj = record['subj']
        cond = record['cond']
        by_source[source][subj].append(record)
        by_subject[subj] += 1
        by_condition[cond] += 1

    # Print header
    msg = "\n" + "="*80
    log(msg, logger)
    msg = "DETAILED MAP COLLECTION BREAKDOWN"
    log(msg, logger)
    msg = "="*80
    log(msg, logger)

    # Summary by source category
    msg = "\nMAPS BY CATEGORY:"
    log(msg, logger)
    total = 0
    for source in sorted(by_source.keys()):
        count = sum(len(maps) for maps in by_source[source].values())
        total += count
        msg = f"  {source:30s}: {count:4d} maps"
        log(msg, logger)
    msg = f"  {'TOTAL':30s}: {total:4d} maps"
    log(msg, logger)

    # Detailed breakdown by source and subject
    msg = "\nDETAILED BREAKDOWN BY SOURCE AND SUBJECT:"
    log(msg, logger)
    for source in sorted(by_source.keys()):
        msg = f"\n  {source}:"
        log(msg, logger)
        for subj in sorted(by_source[source].keys()):
            maps = by_source[source][subj]
            conditions = set(m['cond'] for m in maps)
            msg = f"    {subj}: {len(maps):3d} maps ({len(conditions):2d} conditions)"
            log(msg, logger)
            if logger and logger.verbosity <= logging.INFO:
                # Show first few conditions as examples
                example_conds = sorted(list(conditions))[:5]
                if len(conditions) > 5:
                    example_conds.append(f"... +{len(conditions)-5} more")
                msg = f"           Conditions: {', '.join(example_conds)}"
                log(msg, logger)

    # Summary by subject (across all sources)
    msg = "\nMAPS BY SUBJECT (all sources):"
    log(msg, logger)
    for subj in sorted(by_subject.keys()):
        msg = f"  {subj}: {by_subject[subj]:4d} maps"
        log(msg, logger)

    # Summary by condition (across all sources)
    msg = "\nMAPS BY CONDITION (all sources):"
    log(msg, logger)
    for cond in sorted(by_condition.keys()):
        msg = f"  {cond:30s}: {by_condition[cond]:4d} maps"
        log(msg, logger)

    msg = "="*80 + "\n"
    log(msg, logger)


def ensure_output_file(path, records, existing, logger=None):
    if existing:
        return existing
    data = prepare_output_dict(records, None)
    save_with_retry(path, data, logger)
    return data


def chunk_range(total, chunk_size, chunk_start):
    if chunk_size is None:
        return range(total)
    chunk_size = max(1, chunk_size)
    start = min(max(0, chunk_start), total)
    end = min(total, start + chunk_size)
    return range(start, end)


def collect_hcp_records(path_to_data, subjects):
    records = []
    hcp_dir = op.join(path_to_data, "hcp_results")
    for sub in subjects:
        sub_dir = op.join(hcp_dir, sub)
        for runfolder in list_runfolders(sub_dir):
            eff_dir = op.join(sub_dir, runfolder, "effect_size_maps")
            for fname in list_nii_files(eff_dir):
                records.append(make_hcp_record(sub, runfolder, fname[:-7], op.join(eff_dir, fname)))
    return records


def make_hcp_record(sub, runfolder, cond, path):
    ses = "_".join(runfolder.split("_")[2:])
    # Distinguish FFX (aggregated HCP tasks) from run-level
    source = 'hcp_ffx' if '_ffx' in runfolder else 'hcp_run-level'
    return {'map_path': path, 'raw_path': path, 'subj': sub, 'ses': ses, 'cond': cond, 'source': source}


def load_vector(record, masker, target_affine, target_shape):
    img = image.load_img(record['map_path'])
    # Resample HCP maps (both FFX and run-level) to match target geometry
    if record['source'] in ['hcp_ffx', 'hcp_run-level']:
        img = image.resample_img(img, target_affine=target_affine, target_shape=target_shape, force_resample=True, copy_header=True)
    data = masker.transform(img).ravel()
    del img
    return data


def compute_pair(i, j, records, masker, target_affine, target_shape):
    vec_i = load_vector(records[i], masker, target_affine, target_shape)
    vec_j = load_vector(records[j], masker, target_affine, target_shape)
    coeff = float(np.nan_to_num(np.corrcoef(vec_i, vec_j)[0, 1]))
    del vec_i, vec_j
    gc.collect()
    return coeff


def group_pairs_by_first_index(pairs):
    groups = defaultdict(list)
    for i, j in pairs:
        groups[i].append(j)
    return groups


def compute_pairs_batch(pairs, records, masker, target_affine, target_shape, n_jobs, backend, logger=None):
    if not pairs:
        return {}
    log(f"Computing {len(pairs)} correlations with {n_jobs} workers (backend={backend})...", logger)
    with tqdm_joblib(tqdm(total=len(pairs), desc="Correlations")):
        values = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(compute_pair)(i, j, records, masker, target_affine, target_shape) for i, j in pairs
        )
    return dict(zip(pairs, values))


def init_corr_matrix(size):
    matrix = np.full((size, size), np.nan)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def init_computed_mask(size):
    mask = np.zeros((size, size), dtype=bool)
    np.fill_diagonal(mask, True)
    return mask


def build_metadata(records):
    return {
        'fnames': [r['raw_path'] for r in records],
        'subj': [r['subj'] for r in records],
        'ses': [r['ses'] for r in records],
        'cond': [r['cond'] for r in records],
        'source': [r['source'] for r in records],
        'mapnames': [r['map_path'] for r in records],
    }


def prepare_output_dict(records, existing):
    data = build_metadata(records)
    n = len(records)
    data['corr_matrix'] = init_corr_matrix(n)
    data['computed_mask'] = init_computed_mask(n)
    if existing and existing.get('corr_matrix', np.empty((0, 0))).shape == (n, n):
        data['corr_matrix'] = existing['corr_matrix']
        data['computed_mask'] = existing['computed_mask'] if isinstance(existing.get('computed_mask'), np.ndarray) and existing['computed_mask'].shape == (n, n) else data['computed_mask']
    return data


def collect_pending_pairs_for_indices(path, indices):
    data = load_existing_dict(path)
    mask = data['computed_mask']
    pairs = [(i, j) for i in indices if i < mask.shape[0] for j in range(i + 1, mask.shape[0]) if not mask[i, j]]
    return pairs, data


def merge_updates(path, updates, logger=None):
    if not updates:
        return load_existing_dict(path)
    while True:
        current = load_existing_dict(path)
        for (i, j), value in updates.items():
            current['corr_matrix'][i, j] = current['corr_matrix'][j, i] = value
            current['computed_mask'][i, j] = current['computed_mask'][j, i] = True
        np.fill_diagonal(current['corr_matrix'], 1.0)

        # Prepare metadata for provenance tracking
        n_maps = current['corr_matrix'].shape[0]
        n_computed = int(np.sum(current['computed_mask']) // 2)  # Divide by 2 for symmetry
        n_total = (n_maps * (n_maps - 1)) // 2
        metadata_params = {
            'description': 'Beta map correlation matrix across shinobi and HCP datasets',
            'parameters': {
                'contrasts': CONTRASTS,
                'subjects': SUBJECTS,
                'model': MODEL,
            },
            'subject': None,  # Multi-subject analysis
            'session': None,
            'additional_info': {
                'analysis_type': 'beta_correlations',
                'n_maps': n_maps,
                'n_computed_pairs': n_computed,
                'n_total_pairs': n_total,
                'completion_percentage': (n_computed / n_total * 100) if n_total > 0 else 0,
                'mapnames_count': len(current.get('mapnames', [])),
            }
        }

        save_with_retry(path, current, logger, save_metadata=True, metadata_params=metadata_params)
        return current


def mark_pair(matrix, mask, i, j, value):
    matrix[i, j] = matrix[j, i] = value
    mask[i, j] = mask[j, i] = True


def load_existing_dict(results_path):
    if not op.isfile(results_path):
        return None
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def save_with_retry(path, payload, logger=None, save_metadata=False, metadata_params=None):
    log(f"Saving results to {path}...", logger)
    while True:
        try:
            with open(path, 'wb') as f:
                pickle.dump(payload, f)
            log("Results saved successfully.", logger)

            # Save provenance metadata if requested
            if save_metadata and metadata_params:
                try:
                    metadata = create_metadata(
                        description=metadata_params.get('description', 'Correlation analysis results'),
                        script_path=__file__,
                        output_files=[path],
                        parameters=metadata_params.get('parameters', {}),
                        subject=metadata_params.get('subject'),
                        session=metadata_params.get('session'),
                        additional_info=metadata_params.get('additional_info', {})
                    )
                    metadata_path = path.replace('.pkl', '.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    log(f"Metadata saved to {metadata_path}", logger)
                except Exception as e:
                    log(f"Warning: Failed to save metadata: {e}", logger)

            return
        except OSError as e:
            log("Save failed, retrying in 100ms...", logger)
            if logger:
                logger.error(str(e))
            else:
                print(e)
            time.sleep(0.1)


def save_heatmap(matrix, mapnames, figures_path, logger=None):
    output_dir = op.join(figures_path, "corrmats_withconstant")
    log("Generating heatmap figure...", logger)
    fig, ax = plt.subplots(figsize=(15, 15))
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sbn.heatmap(matrix, xticklabels=mapnames, yticklabels=mapnames, ax=ax, cbar=False, mask=mask, annot=False)
    os.makedirs(output_dir, exist_ok=True)
    destination = op.join(output_dir, "ses-level_corrmat.png")
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    log(f"Heatmap saved to {destination}", logger)


def list_runfolders(sub_dir):
    if not op.isdir(sub_dir):
        return []
    return sorted(f for f in os.listdir(sub_dir) if ("run-" in f or "_ffx" in f))


def list_nii_files(directory):
    if not op.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".nii.gz"))


def submit_slurm_chunks(total_maps, chunk_size, log_dir, verbosity, use_low_level_confs=False, logger=None):
    """Submit SLURM jobs for each chunk of maps."""
    import subprocess

    # Get the project root directory (where slurm/ folder is)
    script_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
    slurm_script = op.join(script_dir, "slurm", "subm_corrmat_chunk.sh")

    if not op.exists(slurm_script):
        msg = f"ERROR: SLURM script not found: {slurm_script}"
        print(msg)
        log(msg, logger)
        return

    num_chunks = (total_maps + chunk_size - 1) // chunk_size  # Ceiling division
    msg = f"Submitting {num_chunks} SLURM jobs for {total_maps} maps (chunk_size={chunk_size})"
    print(msg)
    log(msg, logger)

    # Determine verbosity flag for child jobs
    if verbosity == logging.DEBUG:
        verbose_flag = "-vv"
    elif verbosity == logging.INFO:
        verbose_flag = "-v"
    else:
        verbose_flag = ""

    # Determine low-level confounds flag
    low_level_flag = "--low-level-confs" if use_low_level_confs else ""

    job_ids = []
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        msg = f"Submitting chunk {chunk_idx + 1}/{num_chunks} (maps {chunk_start} to {min(chunk_start + chunk_size - 1, total_maps - 1)})"
        print(msg)
        log(msg, logger)

        try:
            # Pass log_dir, verbosity, low-level-confs flag AND chunk_size to the SLURM script
            result = subprocess.run(
                ["sbatch", slurm_script, str(chunk_start), log_dir or "", verbose_flag, low_level_flag, str(chunk_size)],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract job ID from sbatch output (e.g., "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            msg = f"  → Job ID: {job_id}"
            print(msg)
            log(msg, logger)
        except subprocess.CalledProcessError as e:
            msg = f"ERROR submitting job for chunk {chunk_idx}: {e}"
            print(msg)
            log(msg, logger)
            if logger:
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")

    msg = f"\nSuccessfully submitted {len(job_ids)} jobs: {', '.join(job_ids)}"
    print(msg)
    log(msg, logger)
    msg = f"Monitor with: squeue -u $USER"
    print(msg)
    log(msg, logger)
    msg = f"Cancel all with: scancel {' '.join(job_ids)}"
    print(msg)
    log(msg, logger)


def main():
    args = parse_args()

    # Determine which conditions to use based on --low-level-confs flag
    if args.low_level_confs:
        # Include BOTH game conditions AND low-level features
        contrasts = CONDITIONS + LOW_LEVEL_CONDITIONS
        logger_prefix = "game conditions + low-level features"
    else:
        contrasts = CONDITIONS
        logger_prefix = "game conditions"

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = AnalysisLogger(
        log_name="Correlations",
        log_dir=args.log_dir,
        verbosity=log_level
    )

    try:
        path_to_data = DATA_PATH

        # Determine output path based on whether low-level confounds are used
        processed_dirname = "processed_low-level" if args.low_level_confs else "processed"
        results_path = op.join(DATA_PATH, processed_dirname, 'beta_maps_correlations.pkl')

        log(f"Computing correlations for {logger_prefix}: {contrasts}", logger)
        log("Preparing reference geometry...", logger)
        target_affine, target_shape = get_reference_geometry(path_to_data, SUBJECTS[0])
        log("Fitting shared masker from subject masks...", logger)
        masker = build_masker(SUBJECTS, path_to_data, target_affine, target_shape)
        log("Collecting processed beta maps (session-level variants)...", logger)
        processed_records = collect_processed_records(None, contrasts, MODEL, path_to_data, args.low_level_confs)
        log(f"Discovered {len(processed_records)} session-level maps.", logger)
        log("Collecting subject-level beta maps...", logger)
        subject_records = collect_subject_level_records(contrasts, path_to_data, args.low_level_confs)
        log(f"Discovered {len(subject_records)} subject-level maps.", logger)
        log("Collecting HCP entries...", logger)
        hcp_records = collect_hcp_records(path_to_data, SUBJECTS)
        log(f"Discovered {len(hcp_records)} HCP maps.", logger)
        records = processed_records + subject_records + hcp_records
        if not records:
            log("No maps found.", logger)
            return
        log(f"Total available maps: {len(records)}", logger)

        # Print detailed breakdown of collected maps
        print_detailed_breakdown(records, logger)

        # If --slurm flag is provided, submit batch jobs and exit
        if args.slurm:
            total_maps = len(records)
            # Default to 100 maps per chunk if not specified for SLURM jobs
            chunk_size = args.chunk_size if args.chunk_size is not None else 100
            submit_slurm_chunks(total_maps, chunk_size, args.log_dir, log_level, args.low_level_confs, logger)
            return
        # Initialize output file if it doesn't exist
        existing = load_existing_dict(results_path)
        data = ensure_output_file(results_path, records, existing, logger)

        # Process the assigned chunk
        total_maps = len(records)
        chunk_indices = list(chunk_range(total_maps, args.chunk_size, args.chunk_start))
        if not chunk_indices:
            log("Chunk contains no map indices; exiting.", logger)
            save_heatmap(data['corr_matrix'], data['mapnames'], FIG_PATH, logger)
            return
        log(f"Assigned map indices: {chunk_indices}", logger)
        pairs, latest = collect_pending_pairs_for_indices(results_path, chunk_indices)
        if not pairs:
            log("No pending correlations for this chunk.", logger)
            save_heatmap(latest['corr_matrix'], latest['mapnames'], FIG_PATH, logger)
            return
        updates = compute_pairs_batch(pairs, records, masker, target_affine, target_shape, args.n_jobs, args.backend, logger)
        latest = merge_updates(results_path, updates, logger)
        save_heatmap(latest['corr_matrix'], latest['mapnames'], FIG_PATH, logger)
        
    finally:
        logger.close()


if __name__ == "__main__":
    main()
