import os
import os.path as op
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
import shinobi_fmri.config as config
import matplotlib.pyplot as plt
import itertools
import matplotlib.image as mpimg
import nibabel as nb
from nilearn.plotting import plot_img_on_surf, plot_stat_map
from nilearn.glm import threshold_stats_img
import argparse
from shinobi_fmri.utils.logger import ShinobiLogger
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--subject",
    default=None,
    type=str,
    help="Subject to process",
)
parser.add_argument(
    "-c",
    "--contrast",
    default=None,
    type=str,
    help="Contrast or conditions to compute",
)
parser.add_argument(
    "-v", "--verbose",
    action="count",
    default=0,
    help="Increase verbosity level (e.g. -v for INFO, -vv for DEBUG)",
)
parser.add_argument(
    "--log-dir",
    default=None,
    help="Directory for log files",
)
args = parser.parse_args()


def plot_fullbrain_subjlevel(zmap_fname, output_path, zmap=None, title=None, figpath=None):
    # Format: sub_cond.nii.gz
    # Example: sub-01_HIT.nii.gz
    parts = zmap_fname.split("/")[-1].split("_")
    sub = parts[0]
    cond_name = parts[-1].replace(".nii.gz", "")
    
    if zmap==None:
        zmap = zmap_fname
    
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(zmap, fsaverage.pial_right)    
    # Generate each panel separately
    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        title='Lateral right', vmax=6, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_right, view="lateral",
        output_file=op.join(output_path, f"{sub}_{modeltype}model_{cond_name}_lateral-right.png")
    )
    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        title='Medial right', vmax=6, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_right, view="medial",
        output_file=op.join(output_path, f"{sub}_{modeltype}model_{cond_name}_medial-right.png")
    )
    texture = surface.vol_to_surf(zmap_fname, fsaverage.pial_left)
    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture, hemi='left',
        title='Lateral left', vmax=6, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_left, view="lateral",
        output_file=op.join(output_path, f"{sub}_{modeltype}model_{cond_name}_lateral-left.png")
    )
    fig = plotting.plot_surf_stat_map(
        fsaverage.infl_left, texture, hemi='left',
        title='Medial left', vmax=6, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_left, view="medial",
        output_file=op.join(output_path, f"{sub}_{modeltype}model_{cond_name}_medial-left.png")
    )
    
    # Assemble figure
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(title)
    for idx, view in enumerate(itertools.product(["lateral", "medial"], ["left", "right"])):
        img_fpath = op.join(output_path, f"{sub}_{modeltype}model_{cond_name}_{view[0]}-{view[1]}.png")
        img = mpimg.imread(img_fpath)
        ax = fig.add_subplot(2, 2, idx+1)
        imgplot = plt.imshow(img)
        ax.set_title(f'{view[0]} {view[1]}')
        plt.axis('off')
        plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath)
        plt.close(fig)


def create_viz(sub, cond_name, modeltype, 
               path_to_data=config.DATA_PATH, 
               figures_path=config.FIG_PATH,
               logger=None):
    
    output_path = op.join(figures_path, "subject-level", cond_name, "z_maps", sub)
    os.makedirs(output_path, exist_ok=True)
    folderpath = op.join(path_to_data, "processed", "z_maps", "subject-level")
    
    zmap_fname = op.join(folderpath, cond_name, f"{sub}_{modeltype}model_{cond_name}.nii.gz")
    
    if not op.exists(zmap_fname):
        if logger:
            logger.warning(f"Z-map not found: {zmap_fname}")
        return

    output_path = op.join(figures_path, "subject-level", cond_name, "z_maps", sub)
    os.makedirs(output_path, exist_ok=True)

    if logger:
        logger.info(f"Generating visualizations for {zmap_fname}")

    # Load anat
    anat_fname = op.join(
    path_to_data,
    "cneuromod.processed",
    "smriprep",
    sub,
    "anat",
    f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
    )
    bg_img = nb.load(anat_fname)
    
    # compute thresholds
    uncorrected_map, threshold = threshold_stats_img(zmap_fname, alpha=.001, height_control='fdr')
    cluster_corrected_map, threshold = threshold_stats_img(zmap_fname, alpha=.05, height_control='fdr', cluster_threshold=10)
    
    # unthresholded
    unthresholded_folder = op.join(figures_path, "subject-level", cond_name, "zmap_unthresholded")
    os.makedirs(unthresholded_folder, exist_ok=True)
    plot_img_on_surf(zmap_fname, vmax=6, 
                     title=f'{sub} {cond_name} raw map', 
                     output_file=op.join(unthresholded_folder, f"rawsurf_{sub}_{modeltype}model{cond_name}.png"))
    plot_stat_map(zmap_fname, threshold=3, bg_img=bg_img, vmax=6, display_mode='x', 
                  title=f'{sub} {cond_name} raw map',
                  output_file=op.join(unthresholded_folder, f"slices_{sub}_{modeltype}model{cond_name}.png"))
    html_view = plotting.view_img(zmap_fname, threshold=3, 
                      title=f'{sub} {cond_name} raw map')
    html_view.save_as_html(op.join(unthresholded_folder, f"image_{sub}_{modeltype}model{cond_name}.html"))
    plot_fullbrain_subjlevel(zmap_fname, output_path, zmap=None, 
                             title=f'{sub} {cond_name} raw map',
                             figpath=op.join(unthresholded_folder, f"inflatedsurf_{sub}_{modeltype}model{cond_name}.png"))
    
    if logger:
        logger.debug(f"Saved unthresholded maps to {unthresholded_folder}")

    # thresholded, uncorrected
    thresholded_folder = op.join(figures_path, "subject-level", cond_name, "zmap_thresholded")
    os.makedirs(thresholded_folder, exist_ok=True)
    plot_img_on_surf(uncorrected_map, vmax=6, 
                     title=f'{sub} {cond_name} (FDR<0.001)', 
                     output_file=op.join(thresholded_folder, f"rawsurf_{sub}_{modeltype}model{cond_name}.png"))
    plot_stat_map(uncorrected_map, threshold=3, bg_img=bg_img, vmax=6, display_mode='x', 
                  title=f'{sub} {cond_name} (FDR<0.001)',
                  output_file=op.join(thresholded_folder, f"slices_{sub}_{modeltype}model{cond_name}.png"))
    html_view = plotting.view_img(uncorrected_map, threshold=3, 
                      title=f'{sub} {cond_name} (FDR<0.001)')
    html_view.save_as_html(op.join(thresholded_folder, f"image_{sub}_{modeltype}model{cond_name}.html"))
    plot_fullbrain_subjlevel(zmap_fname, output_path, zmap=uncorrected_map, 
                             title=f'{sub} {cond_name} (FDR<0.001)',
                             figpath=op.join(thresholded_folder, f"inflatedsurf_{sub}_{modeltype}model{cond_name}.png"))
    
    # thresholded, cluster-corrected
    cluster_folder = op.join(figures_path, "subject-level", cond_name, "zmap_cluster")
    os.makedirs(cluster_folder, exist_ok=True)
    plot_img_on_surf(cluster_corrected_map, vmax=6, 
                     title=f'{sub} {cond_name} (FDR<0.05), Clusters > 10vox', 
                     output_file=op.join(cluster_folder, f"rawsurf_{sub}_{modeltype}model{cond_name}.png"))
    plot_stat_map(cluster_corrected_map, threshold=3, bg_img=bg_img, vmax=6, display_mode='x', 
                  title=f'{sub} {cond_name} (FDR<0.05), Clusters > 10vox',
                  output_file=op.join(cluster_folder, f"slices_{sub}_{modeltype}model{cond_name}.png"))
    html_view = plotting.view_img(cluster_corrected_map, threshold=3, 
                      title=f'{sub} {cond_name} (FDR<0.05), Clusters > 10vox')
    html_view.save_as_html(op.join(cluster_folder, f"image_{sub}_{modeltype}model{cond_name}.html"))
    plot_fullbrain_subjlevel(zmap_fname, output_path, zmap=cluster_corrected_map, 
                             title=f'{sub} {cond_name} (FDR<0.05), Clusters > 10vox',
                             figpath=op.join(cluster_folder, f"inflatedsurf_{sub}_{modeltype}model{cond_name}.png"))

    if logger:
        logger.log_computation_success("Viz_subject-level", zmap_fname)


if __name__ == "__main__":
    COND_LIST = ['HIT', 'JUMP', 'DOWN', 'HealthGain', 'HealthLoss', 
                 'Kill', 'LEFT', 'RIGHT', 'UP', 
                 'HIT+JUMP', 
                 'RIGHT+LEFT+DOWN']
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = config.SUBJECTS

    if args.contrast is not None:
        contrasts = [args.contrast]
    else:
        contrasts = COND_LIST

    # Determine verbosity
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    # Initialize logger
    logger = ShinobiLogger(
        log_name="Viz_subject",
        log_dir=args.log_dir,
        verbosity=log_level
    )
    
    try:
        for sub in subjects:
            for cond_name in contrasts:
                for modeltype in ["full", "simple", "intermediate"]:
                    try:
                        folderpath = op.join(config.DATA_PATH, "processed", "z_maps", "subject-level")
                        zmap_fname = op.join(folderpath, cond_name, f"{sub}_{modeltype}model_{cond_name}.nii.gz")
                        if op.exists(zmap_fname):
                            logger.info(f"Creating viz for {sub} {cond_name} {modeltype}")
                            create_viz(sub, cond_name, modeltype, logger=logger)
                    except Exception as e:
                        logger.log_computation_error(f"Viz_{sub}_{cond_name}", e)
    finally:
        logger.close()
