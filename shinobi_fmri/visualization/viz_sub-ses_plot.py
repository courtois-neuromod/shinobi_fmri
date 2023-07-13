import nilearn
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import shinobi_behav
import nibabel as nib
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.gridspec as gridspec
import math
import glob


def create_colormap():
    # Get the colormap
    cmap = nilearn_cmaps['cold_hot']

    # Create a list of colors from the colormap
    colors = cmap(np.linspace(0, 1, cmap.N))

    # Find the indices corresponding to -3 and 3 on the color scale
    lower_bound = int(64)  # -3 corresponds to this index
    upper_bound = int(192)  # 3 corresponds to this index

    # Set the colors in those indices to grey
    colors[lower_bound:upper_bound, :] = [0.5, 0.5, 0.5, 1]  # RGBA for grey

    # Create a new colormap with these colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list('new_cmap', colors)

    # Create a normalization object that maps the data range (-6, 6) to the range (0, 1)
    norm = mcolors.Normalize(vmin=-6, vmax=6)

    # Now we can create a ScalarMappable with the new colormap and normalization, and use it to create a colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=new_cmap)
    mappable.set_array([])  # Needed for the colorbar function

    # Create a figure and add the colorbar to it
    fig, ax = plt.subplots(figsize=(1, 6))
    plt.colorbar(mappable, ax=ax, orientation='vertical')
    ax.remove()
    return fig, ax, mappable


def create_pdf_with_images(image_folder, pdf_filename):
    """Create a PDF with all the images in a folder

    Args:
        image_folder (str): Path to the folder containing the images.
        pdf_filename (str): Path to the PDF file to create.
    Returns:
        None
    """
    images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')])

    c = canvas.Canvas(pdf_filename)
    size = Image.open(images[0]).size
    c.setPageSize(size)
    

    for image_path in images:
        print(image_path)
        c.drawImage(image_path, 0, 0, width=size[0], height=size[1])  # Draws image on the canvas
        c.showPage()  # Ends the current page and starts a new one

    c.save()  # Save the PDF

def plot_inflated_zmap(img, save_path=None, title=None, colorbar=True, vmax=6, threshold=0.9, dpi=300):
    """Plot a seed-based connectivity surface map and save the image.
    Args:
        img (str): Path to the image to plot.
        save_path (str): Path to save the image to.
        title (str): Title of the image.
        colorbar (bool): Whether to include a colorbar.
        cmap (str): Colormap to use.
        vmax (float): Maximum value for the colormap.
        threshold (float): Threshold for the image.
        dpi (int): DPI of the image.
    Returns:
        None
    """
    if threshold is not None:
        img_data = nib.load(img).get_fdata() if isinstance(img, str) else img.get_fdata()
        img_data = img_data[img_data != 0].flatten()
        thres_val = np.sort(img_data)[int(img_data.shape[0] * threshold)]
    else:
        thres_val = None
    plt.rcParams['figure.dpi'] = dpi
    
    plotting.plot_img_on_surf(
        img,
        views=["lateral", "medial"],
        hemispheres=["left", "right"],
        inflate=True,
        colorbar=colorbar,
        threshold=3,#thres_val,
        vmax=vmax,
        symmetric_cbar=False,
        output_file=save_path,
        title=title
    )

def create_all_images(subject, condition, fig_folder):
    '''Create all images for a given subject and condition.
    Args:
        subject (str): Subject ID.
        condition (str): Condition (contrast) to process.
        fig_folder (str): Path to save the images to.
    Returns:
        None
    '''
    # Create images
    ## Make subject level z_maps
    print(f"Create images for {subject} {condition}")
    sublevel_zmap_path = os.path.join(shinobi_behav.DATA_PATH, 
                                      "processed",
                                      "z_maps",
                                      "subject-level",
                                      condition, 
                                      f"{subject}_simplemodel_{condition}.nii.gz")
    sublevel_save_path = os.path.join(fig_folder,
                                      f"{subject}_{condition}.png")
    if not os.path.isfile(sublevel_save_path):
        if os.path.isfile(sublevel_zmap_path):
            plot_inflated_zmap(sublevel_zmap_path, save_path=sublevel_save_path, title=f"{subject}", colorbar=False)
        # If no file, then create an empty image
        else:
            allfigs_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", 
                "reports", "figures", "full_zmap_plot", subject)
            other_images = glob.glob(os.path.join(allfigs_folder, "*", "*.png"))
            model_image = other_images[0]
            img_size = Image.open(model_image).size
            missing_img = Image.new('RGB', img_size, color = (255, 255, 255))
            d = ImageDraw.Draw(missing_img)
            #d.text((img_size[0]//2,img_size[1]//2), "Missing", fill=(0,0,0))
            missing_img.save(sublevel_save_path)
        
    ## Make ses level z_maps
    ses_list = sorted(os.listdir(os.path.join(shinobi_behav.DATA_PATH, "shinobi", subject)))
    for session in ses_list:
        
        zmap_path = os.path.join(shinobi_behav.DATA_PATH, 
                                          "processed",
                                          "z_maps",
                                          "ses-level",
                                          condition, 
                                          f"{subject}_{session}_simplemodel_{condition}.nii.gz")
        save_path = os.path.join(fig_folder,
                                f"{subject}_{session}_{condition}.png")
        if not os.path.isfile(save_path):
            if os.path.isfile(zmap_path):
                plot_inflated_zmap(zmap_path, save_path=save_path, title=f"{session}", colorbar=False)
            # If no file, then create an empty image
            else:
                allfigs_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", 
                    "reports", "figures", "full_zmap_plot", subject)
                other_images = glob.glob(os.path.join(allfigs_folder, "*", "*.png"))
                model_image = other_images[0]
                img_size = Image.open(model_image).size
                missing_img = Image.new('RGB', img_size, color = (255, 255, 255))
                d = ImageDraw.Draw(missing_img)
                #d.text((img_size[0]//2,img_size[1]//2), "Missing", fill=(0,0,0))
                missing_img.save(save_path)

def make_annotation_plot(condition, save_path):
    '''Make a 8*4 plot for one annotation, with all the subjects on the same page
    '''
    images = []
    for idx_subj, subject in enumerate(shinobi_behav.SUBJECTS):
        print(f"Processing {subject}")
        fig_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", 
                        "reports", "figures", "full_zmap_plot", subject, condition)
        # obtain top 4 sesmaps
        ses_list = sorted(os.listdir(os.path.join(shinobi_behav.DATA_PATH, "shinobi", subject)))
        sesmap_vox_above_thresh = []
        sesmap_name = []
        for session in ses_list:
            try:
                print(f"Loading {session}")
                zmap_path = os.path.join(shinobi_behav.DATA_PATH, 
                                                "processed",
                                                "z_maps",
                                                "ses-level",
                                                condition, 
                                                f"{subject}_{session}_simplemodel_{condition}.nii.gz")
                img = nib.load(zmap_path).get_fdata()
                img_vox_above_thresh = [1 if abs(x) > 3 else 0 for x in img.flatten()]
                nb_vox_above_thresh = sum(img_vox_above_thresh)
                sesmap_vox_above_thresh.append(nb_vox_above_thresh)
                sesmap_name.append(session)
            except FileNotFoundError:
                print("File not found")
        top4_idx = np.argsort(sesmap_vox_above_thresh)[-4:]
        top4_names = np.sort(np.array(sesmap_name)[top4_idx])

        # Add images to the list
        # subject-level image
        subj_images = [np.array(Image.open(os.path.join(fig_folder, f"{subject}_{condition}.png")))]
        for sesname in top4_names:
            subj_images.append(np.array(Image.open(os.path.join(fig_folder,
                    f"{subject}_{sesname}_{condition}.png"))))
        images.append(subj_images)
        
    # Make figure
    # Create figure with specific size
    fig = plt.figure(figsize=(16,8), dpi=300) # You can adjust these values
    #fig.patch.set_facecolor('grey')

    # Create a 8 by 4 grid
    gs = fig.add_gridspec(4, 9)

    #ax_lines = fig.add_subplot(gs[:,:8])
    #ax_lines.axvline(x=0.5, color='grey')

    # Draw a horizontal line in the middle of the graph, 
    # x range will be automatically adjusted to the graph dimensions
    #ax_lines.axhline(y=0.5, color='grey')
    #ax_lines.axis('off')

    for idx_subj, subject in enumerate(shinobi_behav.SUBJECTS):
        # Create a larger subplot for the first image
        if idx_subj == 0:
            ax1 = fig.add_subplot(gs[0:2, 0:2])
        elif idx_subj == 1:
            ax1 = fig.add_subplot(gs[0:2, 4:6])
        elif idx_subj == 2:
            ax1 = fig.add_subplot(gs[2:4, 0:2])
        elif idx_subj == 3:
            ax1 = fig.add_subplot(gs[2:4, 4:6])
        # Display the first image
        ax1.imshow(images[idx_subj][0])
        ax1.axis('off')  # To remove axes

        # Display the rest of the images

        for i in range(len(images[idx_subj][1:])):
            if i == 0:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+0,((idx_subj%2)*4)+2])
            elif i == 1:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+0,((idx_subj%2)*4)+3])
            elif i == 2:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+1,((idx_subj%2)*4)+2])
            elif i == 3:
                ax = fig.add_subplot(gs[(math.floor(idx_subj/2)*2)+1,((idx_subj%2)*4)+3])
            ax.imshow(images[idx_subj][i+1])
            ax.axis('off')  # To remove axes

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f"{condition}", fontsize=18, x=0.4444)
    _,_,cmap = create_colormap()
    # Define the axes for the colorbar in the 8th column of the GridSpec
    inner_gs = gridspec.GridSpecFromSubplotSpec(4, 8, subplot_spec=gs[:, 8])
    cbar_ax = fig.add_subplot(inner_gs[1:3, 0])

    # Create the colorbar in the defined axes
    plt.colorbar(cmap, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig(save_path)


if __name__ == "__main__":

    output_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", "reports", "figures", "full_zmap_plot", "annotations")
    os.makedirs(output_folder, exist_ok=True)

    for condition in ['Kill', 'HealthLoss', 'JUMP', 'HIT', 'DOWN', 'LEFT', 'RIGHT', 'UP']:#, 'lvl1', 'lvl4', 'lvl5']:
        for subject in shinobi_behav.SUBJECTS:
            fig_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", 
                                    "reports", "figures", "full_zmap_plot", subject, condition)
            os.makedirs(fig_folder, exist_ok=True)

            create_all_images(subject, condition, fig_folder)
        save_path = os.path.join(output_folder, f"annotations_plot_{condition}.png")
        make_annotation_plot(condition, save_path)
    for lvl in []:#'lvl1', 'lvl4', 'lvl5']:
        for condition in ['Kill', 'HealthLoss', 'JUMP', 'HIT', 'DOWN', 'LEFT', 'RIGHT', 'UP']:
            condition = f"{condition}X{lvl}"
            for subject in shinobi_behav.SUBJECTS:
                fig_folder = os.path.join("/home/hyruuk/projects/def-pbellec/hyruuk/shinobi_fmri", 
                                        "reports", "figures", "full_zmap_plot", subject, condition)
                os.makedirs(fig_folder, exist_ok=True)

                create_all_images(subject, condition, fig_folder)
            save_path = os.path.join(output_folder, f"annotations_plot_{condition}.png")
            make_annotation_plot(condition, save_path)
    create_pdf_with_images(output_folder, os.path.join(output_folder, 'inflated_zmaps_by_annot.pdf'))