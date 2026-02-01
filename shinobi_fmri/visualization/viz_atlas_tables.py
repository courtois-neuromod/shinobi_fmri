
import os
import os.path as op
import glob
import argparse
import pandas as pd
import numpy as np
import re

# Patch nilearn._utils for atlasreader compatibility
try:
    import nilearn._utils
    from nilearn._utils.niimg_conversions import check_niimg
    nilearn._utils.check_niimg = check_niimg
except ImportError:
    pass

from atlasreader import create_output
from shinobi_fmri.config import DATA_PATH, FIG_PATH

def parse_bids_filename(filename):
    """
    Parse a BIDS-like filename to extract subject and contrast/annotation.
    Example: sub-01_task-shinobi_contrast-DOWN_stat-z.nii.gz
    """
    # Remove extension
    base = filename.split('.')[0]
    if base.endswith('_clusters'):
        base = base.replace('_clusters', '')
    
    parts = base.split('_')
    
    subject = None
    contrast = None
    
    for part in parts:
        if part.startswith('sub-'):
            subject = part
        elif part.startswith('contrast-'):
            contrast = part.replace('contrast-', '')
            
    # Fallback if simple split doesn't work or if structure is different
    if subject is None and parts:
        subject = parts[0] # Assume first part is subject
        
    return subject, contrast

def generate_atlas_tables(input_dir, output_dir, cluster_extent=5, voxel_thresh=3, direction='both', use_raw_maps=False, overwrite=False):
    """
    Generate atlas tables for z-maps.

    Args:
        input_dir: Directory containing subject folders (e.g. processed/subject-level).
        output_dir: Directory to save the output CSV files.
        cluster_extent: Minimum cluster size in voxels.
        voxel_thresh: Voxel threshold for significance.
        direction: Direction of the contrast ('both', 'pos', 'neg').
        use_raw_maps: If True, use raw z-maps instead of corrected maps (default: False, use corrected).
        overwrite: If True, overwrite existing files.
    """

    if not op.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Scanning for z-maps in {input_dir}...")
    print(f"Using {'raw uncorrected' if use_raw_maps else 'corrected'} z-maps")

    # Check if input_dir exists
    if not op.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Find subject directories
    subjects = sorted([d for d in os.listdir(input_dir) if d.startswith('sub-') and op.isdir(op.join(input_dir, d))])

    if not subjects:
        print(f"No subject directories found in {input_dir}")
        return

    count = 0
    for sub in subjects:
        # Check for z_maps directory
        zmaps_path = op.join(input_dir, sub, 'z_maps')
        if not op.exists(zmaps_path):
            continue

        # Use corrected or raw maps based on flag (default: corrected)
        if use_raw_maps:
            # Look for raw z-maps only (no desc-corrected in filename)
            zmaps_list = glob.glob(op.join(zmaps_path, '*_stat-z.nii.gz'))
            # Exclude corrected maps from the list
            zmaps_list = [z for z in zmaps_list if 'desc-corrected' not in z]
        else:
            # Look for corrected z-maps first (default)
            zmaps_list = glob.glob(op.join(zmaps_path, '*_desc-corrected_stat-z.nii.gz'))
            # Fall back to raw maps if no corrected maps found
            if not zmaps_list:
                zmaps_list = glob.glob(op.join(zmaps_path, '*_stat-z.nii.gz'))
                zmaps_list = [z for z in zmaps_list if 'desc-corrected' not in z]
                if zmaps_list:
                    print(f"  Note: Using raw maps for {sub} (corrected maps not found)")
        
        for zmap in zmaps_list:
            
            output_csv_name = op.basename(zmap).replace('.nii.gz', '_clusters.csv')
            output_csv_path = op.join(zmaps_path, output_csv_name)
            
            if op.exists(output_csv_path) and not overwrite:
                # print(f"  Skipping {op.basename(zmap)} (clusters file exists)")
                continue
            
            print(f"Processing {sub} / {op.basename(zmap)}...")
            try:
                create_output(zmap, cluster_extent=cluster_extent, voxel_thresh=voxel_thresh, direction=direction)
                count += 1
            except Exception as e:
                print(f"  Error processing {zmap}: {e}")

    if count == 0:
        print("No new tables generated (or files already existed). Proceeding to aggregation.")

    # Aggregate results
    print("Aggregating cluster tables...")
    # Pattern: input_dir/sub-*/z_maps/*_clusters.csv
    tables_list = glob.glob(op.join(input_dir, 'sub-*', 'z_maps', '*clusters.csv'))

    if not tables_list:
        print("No cluster tables found.")
        return

    tables_df = []
    for table in tables_list:
        try:
            df = pd.read_csv(table)
            filename = op.basename(table)

            subject, annotation = parse_bids_filename(filename)

            if not annotation:
                # Fallback: try to guess or just use filename
                annotation = filename

            df['annotation'] = annotation
            df['subject'] = subject

            # Select 3 biggest clusters by cluster_mean if available
            if 'cluster_mean' in df.columns:
                 df = df.sort_values(by='cluster_mean').tail(3)

            tables_df.append(df)
        except Exception as e:
            print(f"Error reading {table}: {e}")

    if not tables_df:
        print("No data to aggregate.")
        return

    tables_df = pd.concat(tables_df, ignore_index=True).astype(str)

    cols_to_drop = ['desikan_killiany', 'harvard_oxford']
    tables_df = tables_df.drop(columns=[c for c in cols_to_drop if c in tables_df.columns])

    def concatenate_strings(series):
        return "; ".join(series)

    agg_dict={}
    for col in tables_df.columns:
        if col not in(['annotation', 'subject']):
            agg_dict[col] = concatenate_strings

    tables_df_grouped = tables_df.groupby(['annotation', 'subject']).agg(agg_dict)

    # Format cluster info string
    req_cols = ['cluster_id', 'peak_x', 'peak_y', 'peak_z', 'cluster_mean', 'volume_mm']
    if all(col in tables_df_grouped.columns for col in req_cols):
        tables_df_grouped['cluster (id / x / y / z / mean / volume)'] = (
            tables_df_grouped['cluster_id'] + ' / ' +
            tables_df_grouped['peak_x'] + ' / ' +
            tables_df_grouped['peak_y'] + ' / ' +
            tables_df_grouped['peak_z'] + ' / ' +
            tables_df_grouped['cluster_mean'] + ' / ' +
            tables_df_grouped['volume_mm']
        )
        tables_df_grouped.drop(columns=req_cols, inplace=True)

    # Add suffix to indicate corrected vs raw maps
    map_type = "raw" if use_raw_maps else "corrected"
    output_file = op.join(output_dir, f'cluster_tables_extent-{cluster_extent}_thresh-{voxel_thresh}_{direction}_{map_type}.csv')
    print(f"Saving aggregated table to {output_file}")
    tables_df_grouped.to_csv(output_file)

    if 'aal' in tables_df_grouped.columns:
        print("Generating occurrence table...")
        generate_occurrence_table(tables_df_grouped, output_dir, cluster_extent, voxel_thresh, direction, use_raw_maps)

def generate_occurrence_table(tables_df_grouped, output_dir, cluster_extent, voxel_thresh, direction, use_raw_maps=False):
    # Extract labels from the aggregated dataframe
    data = []

    for (annotation, subject), content in tables_df_grouped.iterrows():
        if not isinstance(content['aal'], str):
            continue

        for label_entry in content['aal'].split('; '):
            if '%' in label_entry:
                try:
                    # Parse string like "33.22% Parietal_Sup_L"
                    label_clean = label_entry.split('%')[1].strip()
                    data.append({
                        'label': label_clean,
                        'annotation': annotation,
                        'subject': subject
                    })
                except IndexError:
                    continue

    labels_df = pd.DataFrame(data)

    if labels_df.empty:
        print("No labels found to analyze.")
        return

    # Count occurrences of each label
    label_counts = labels_df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'occurences']

    # Function to summarize presence
    def get_presence(label, df, col):
        unique_vals = sorted(df[df['label'] == label][col].unique())
        return "; ".join(unique_vals) + "; "

    label_counts['present in annotation'] = label_counts['label'].apply(
        lambda x: get_presence(x, labels_df, 'annotation')
    )
    label_counts['present in subject'] = label_counts['label'].apply(
        lambda x: get_presence(x, labels_df, 'subject')
    )

    # Add suffix to indicate corrected vs raw maps
    map_type = "raw" if use_raw_maps else "corrected"
    output_file = op.join(output_dir, f'occurence_df_extent-{cluster_extent}_thresh-{voxel_thresh}_{direction}_{map_type}.csv')
    print(f"Saving occurrence table to {output_file}")
    label_counts.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate atlas tables for z-maps.")
    # UPDATED DEFAULT PATH
    parser.add_argument("--input-dir", type=str, default=op.join(DATA_PATH, "processed", "subject-level"), help="Directory containing subject folders.")
    parser.add_argument("--output-dir", type=str, default=op.join("reports", "tables"), help="Directory to save output tables.")
    parser.add_argument("--cluster-extent", type=int, default=5, help="Minimum cluster size in voxels.")
    parser.add_argument("--voxel-thresh", type=float, default=3.0, help="Voxel threshold for significance.")
    parser.add_argument("--direction", type=str, default="both", choices=["both", "pos", "neg"], help="Direction of the contrast.")
    parser.add_argument("--use-raw-maps", action="store_true", help="Use raw z-maps instead of corrected maps (default: use corrected maps).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cluster files.")
    parser.add_argument("--no-low-level", action="store_true", help="Exclude low-level features from visualization (default: False, low-level features are included).")

    args = parser.parse_args()

    # Always use processed/ directory (low-level features are now default)
    # The --no-low-level flag doesn't change the directory since all data is in processed/

    generate_atlas_tables(args.input_dir, args.output_dir, args.cluster_extent, args.voxel_thresh, args.direction, args.use_raw_maps, args.overwrite)
