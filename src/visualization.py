"""
3D Slicer, FAST
"""
import scipy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import ListedColormap
from preprocessing import get_2d_mask, get_3d_mask, get_largest_connected_component, apply_mask, apply_bilateral_filter, apply_bilateral_filter_sitk, apply_window
import imageio
from skimage import measure
from icecream import ic
import numpy as np
import os

def render_volume_slices(volume, cmap='grey', n_rows=None, ax_size=4, v_min=None, v_max=None):
    """
    Renders all slices of a 3D volume with no gaps between images.
    
    Parameters:
    - volume: 3D numpy array representing the volume
    - cmap: colormap for the images (default: 'grey')
    - figsize: size of the figure (default: (12, 12))
    
    Returns:
    - fig: matplotlib figure object
    - axes: 2D array of matplotlib axes objects
    """
    depth = len(volume)
    if n_rows is not None:
        rows = n_rows
    else:
        rows = int(np.sqrt(depth))
    cols = int(np.ceil(depth / rows))
    figsize = (cols * ax_size, rows * ax_size)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(wspace=0, hspace=0)
    if v_min is None:
        v_min = np.min(volume)
    if v_max is None:
        v_max = np.max(volume)
    for i, ax in enumerate(axes.flat):
        if i < depth:
            ax.imshow(volume[i], cmap=cmap, aspect='equal', vmin=v_min, vmax=v_max)
        ax.axis('off')
    
    plt.tight_layout()
    fig.patch.set_facecolor('black')
    plt.show(block=True)
    return fig, axes
def scroll_through_all_slices(volume_seq: np.ndarray, title="", show=True, cmap='grey'):
    t, y, z, x = volume_seq.shape
    # Reshape the 4D volume sequence into a 3D array
    volume = volume_seq.reshape((t * y, z, x))
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Set the title
    plt.title(title)
    
    # Display the initial slice
    img = ax.imshow(volume[0], cmap=cmap)
    plt.colorbar(img)
    
    # Create the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=0, valstep=1)
    
    # Update function for the slider
    def update(val):
        slice_index = int(slider.val)
        img.set_data(volume[slice_index])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    if show:
        plt.show()
    
    return fig, ax

def interactive_plot(volume_seq, title="", cmap='grey', show=True, windowing_params=None):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    if windowing_params is not None:
        # Getting intensity interval from windowing params
        vmin, vmax = windowing_params[0] - windowing_params[1] / 2, windowing_params[0] + windowing_params[1] / 2
        image = ax.imshow(apply_window(volume_seq[0, volume_seq.shape[1]//2], *windowing_params), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        vmin = np.min(volume_seq)
        vmax = np.max(volume_seq) 
        image = ax.imshow(volume_seq[0, 0], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)

    ax_slice_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])

    time_slider = Slider(ax_time_slider, 'Time', 0, volume_seq.shape[0] - 1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, volume_seq.shape[1] - 1, valinit=0, valstep=1)
    scrolling_slider = [time_slider]
    plt.colorbar(image, ax=ax)

    def update(val, scrolling_slider, slider):
        if windowing_params is not None:
            image.set_data(apply_window(volume_seq[int(time_slider.val), int(slice_slider.val)], *windowing_params))
        else:
            image.set_data(volume_seq[int(time_slider.val), int(slice_slider.val)])
        fig.canvas.draw_idle()
        scrolling_slider[0] = slider

    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))

    def on_scroll(event):
        if event.button == 'up':
            scrolling_slider[0].set_val(min(scrolling_slider[0].val + scrolling_slider[0].valstep, scrolling_slider[0].valmax))
        elif event.button == 'down':
            scrolling_slider[0].set_val(max(scrolling_slider[0].val - scrolling_slider[0].valstep, scrolling_slider[0].valmin))
        fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes == ax:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, ax

def interactive_plot_cycle_from_folder(data_folder="Experiments/Data", windowing_params=None, json_path="volumes.json", cmap='grey'):
    """
    Cycle through npy volume files stored in the provided folder. Each volume is assumed 
    to be a 4D numpy array with shape (time, slice, height, width).

    The plot displays the current volume with interactive sliders for time and slice.
    A key press listener is attached so that:
      - Pressing Space adds the current volume to "exclude" and cycles to next volume.
      - Pressing Shift adds the current volume to "re-register" and cycles to next volume.
      - Pressing Enter saves the current volume title (derived from the filename, without extension)
        to a JSON file. The title is always added to the "checked" category and, if the Space key was pressed,
        it is also added to "exclude". In addition, if volumes have already been processed 
        (i.e. present in the JSON file under "checked"), those files will be skipped.
      - Pressing Backspace reverses the last scan: it deletes the last scan from the JSON data and 
        changes the plot to the previous scan.
      Instead of closing the window, the function updates the figure with the next volume.

    Parameters:
      data_folder: str, path to the folder containing npy volume files (default "Experiments/Data")
      windowing_params: parameters passed to apply_window (default: None)
      json_path: str, path to the JSON file to store volume names (default "volumes.json")
      
    Usage:
      interactive_plot_cycle_from_folder(data_folder="@Data")
    """

    plt.ion()  # Ensure interactive mode is on.

    # Load JSON data early; we use the same dict throughout.
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except Exception:
            json_data = {}
    else:
        json_data = {}
    json_data.setdefault("checked", [])
    json_data.setdefault("exclude", [])
    json_data.setdefault("re-register", [])

    # Get a sorted list of npy files and filter out volumes already checked.
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    files = [f for f in all_files if os.path.splitext(f)[0] not in json_data["checked"]]
    # files = [f for f in all_files if os.path.splitext(f)[0] in json_data["exclude"]]
    if not files:
        print("No new volumes to display; all volumes have been checked.")
        return None, None

    current_idx = 0

    def load_volume(idx):
        filepath = os.path.join(data_folder, files[idx])
        volume = np.load(filepath)
        title = os.path.splitext(files[idx])[0]
        return volume, title

    current_volume, current_title = load_volume(current_idx)

    # Create the figure and initial axes.
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    if windowing_params is not None:
        img_data = apply_window(current_volume[0, 0], *windowing_params)
    else:
        img_data = current_volume[0, 0]
    image = ax.imshow(img_data, cmap=cmap)
    ax.set_title(current_title)
    plt.colorbar(image, ax=ax)

    # Create slider axes for time and slice.
    ax_slice_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(ax_time_slider, 'Time', 0, current_volume.shape[0] - 1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, current_volume.shape[1] - 1, valinit=0, valstep=1)
    scrolling_slider = [time_slider]  # Mutable container for the slider.

    def update_image():
        """Update the image based on the current slider values."""
        t = int(time_slider.val)
        s = int(slice_slider.val)
        if windowing_params is not None:
            new_data = apply_window(current_volume[t, s], *windowing_params)
        else:
            new_data = current_volume[t, s]
        image.set_data(new_data)
        fig.canvas.draw_idle()

    def update(val, scrolling_slider, slider):
        update_image()
        scrolling_slider[0] = slider

    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))

    def on_scroll(event):
        if event.inaxes == ax_time_slider:
            new_val = min(time_slider.val + event.step, time_slider.valmax)
            time_slider.set_val(new_val)
        elif event.inaxes == ax_slice_slider:
            new_val = min(slice_slider.val + event.step, slice_slider.valmax)
            slice_slider.set_val(new_val)
        fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes == ax:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    def on_key_press(event):
        nonlocal current_idx, current_volume, current_title, json_data
        if event.key in [' ', 'space']:
            # Add current volume to checked and exclude lists
            if current_title not in json_data["checked"]:
                json_data["checked"].append(current_title)
            if current_title not in json_data["exclude"]:
                json_data["exclude"].append(current_title)
            
            # Save JSON data
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            print(f"Added '{current_title}' to exclude and checked lists")

            # Move to next volume
            current_idx += 1
            if current_idx < len(files):
                current_volume, current_title = load_volume(current_idx)
                ax.set_title(current_title)
                update_image()
            else:
                print("No more volumes to display.")

        elif event.key == 'shift':
            # Add current volume to checked and register lists
            if current_title not in json_data["checked"]:
                json_data["checked"].append(current_title)
            if current_title not in json_data["re-register"]:
                json_data["re-register"].append(current_title)
            if current_title in json_data["exclude"]:
                json_data["exclude"].remove(current_title)
            # Save JSON data
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            print(f"Added '{current_title}' to re-register and checked lists")

            # Move to next volume
            current_idx += 1
            if current_idx < len(files):
                current_volume, current_title = load_volume(current_idx)
                ax.set_title(current_title)
                update_image()
            else:
                print("No more volumes to display.")

        elif event.key == 'backspace':
            if current_idx == 0:
                print("Already at the first volume; cannot reverse further.")
                return
            # Reverse action:
            current_idx -= 1
            # Determine the title from the previous file.
            prev_file = files[current_idx]
            prev_title = os.path.splitext(prev_file)[0]
            # Remove the previous volume from JSON data if present.
            if prev_title in json_data["checked"]:
                json_data["checked"].remove(prev_title)
                print(f"Removed '{prev_title}' from checked.")
            if prev_title in json_data["exclude"]:
                json_data["exclude"].remove(prev_title)
                print(f"Removed '{prev_title}' from exclude.")
            if prev_title in json_data["re-register"]:
                json_data["re-register"].remove(prev_title)
                print(f"Removed '{prev_title}' from re-register.")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            # Load the previous volume.
            current_volume, current_title = load_volume(current_idx)
            ax.set_title(current_title)
            update_image()
            print(f"Reverted to volume: {current_title}")
        elif event.key == 'enter':
            # Add the current volume's title to json_data.
            if current_title not in json_data["checked"]:
                json_data["checked"].append(current_title)
            if current_title in json_data["re-register"]:
                json_data["re-register"].remove(current_title)
            if current_title in json_data["exclude"]:
                json_data["exclude"].remove(current_title)
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            print(f"Saved '{current_title}' to checked list")

            # Move to the next volume and update the same window.
            current_idx += 1
            if current_idx < len(files):
                current_volume, current_title = load_volume(current_idx)
                ax.set_title(current_title)
                update_image()
            else:
                print("No more volumes to display.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show(block=True)
    return fig, ax


def multiple_interactive_plots(volume_seqs, titles=None, plotting_function=interactive_plot, cmap='grey'):
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(len(volume_seqs))]
    
    figures = []
    for volume_seq, title in zip(volume_seqs, titles):
        fig, ax = plotting_function(volume_seq, title, cmap=cmap, show=False)
        figures.append(fig)
    
    plt.show(block=False)
    
    input("Press Enter to close all windows...")
    for fig in figures:
        plt.close(fig)
    plt.ioff()  # Turn off interactive mode

def multi_vol_seq_iplot(volume_seqs, titles=None, nrows=None, cmap='grey'):
    """
    Renders multiple interactive plots in one scene with two unified sliders
    controlling all plots
    volume_seqs: iterable of 4D volume sequences.
    """
    plt.ion()  # Turn on interactive mode
    num_volumes = len(volume_seqs)

    if nrows is None:
        nrows = int(num_volumes ** .5)
    ncols = (num_volumes + nrows - 1) // nrows

    fig, axes = plt.subplots(nrows, 
                             ncols, 
                             figsize=(5*ncols, 5*nrows),
                             squeeze=True,
                            #  sharex='all',
                            #  sharey='all'
                             )
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.1, hspace=0.4)

    ic(axes)
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(num_volumes)]

    images = []
    for i, (volume_seq, title) in enumerate(zip(volume_seqs, titles)):
        # row, col = divmod(i, ncols)
        ax = axes[i] #if nrows == 1 else axes[row, col]
        image = ax.imshow(volume_seq[0, 0], cmap=cmap)
        ax.set_title(title)
        plt.colorbar(image, ax=ax)
        images.append(image)

    ax_slice_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    ax_time_slider = plt.axes([0.25, 0.05, 0.5, 0.03])

    max_time = max(vol.shape[0] for vol in volume_seqs) - 1
    max_slice = max(vol.shape[1] for vol in volume_seqs) - 1

    time_slider = Slider(ax_time_slider, 'Time', 0, max_time, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, max_slice, valinit=0, valstep=1)

    def update(val):
        for i, volume_seq in enumerate(volume_seqs):
            t = min(int(time_slider.val), volume_seq.shape[0] - 1)
            s = min(int(slice_slider.val), volume_seq.shape[1] - 1)
            images[i].set_data(volume_seq[t, s])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    slice_slider.on_changed(update)

    def on_scroll(event):
        if event.inaxes == ax_time_slider:
            time_slider.set_val(min(max(time_slider.val + event.step, time_slider.valmin), time_slider.valmax))
        elif event.inaxes == ax_slice_slider:
            slice_slider.set_val(min(max(slice_slider.val + event.step, slice_slider.valmin), slice_slider.valmax))
        else:
            for ax in axes.flat:
                if event.inaxes == ax:
                    slice_slider.set_val(min(max(slice_slider.val + event.step, slice_slider.valmin), slice_slider.valmax))
                    break
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show(block=True)

def interactive_plot_with_threshold(volume_seq, title="", cmap='grey', show=True, min_thresh=-40, max_thresh=120, apply_window=True):

    """Like interactive_plot but with additional threshold sliders that show a binary mask"""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9, wspace=0.3)
    
    # Original image on left
    if apply_window:
        image = ax1.imshow(np.clip(volume_seq[0, 0], a_min=-40, a_max=120), cmap=cmap)
    else:
        image = ax1.imshow(volume_seq[0, 0], cmap=cmap)
    ax1.set_title(f"{title} - Original")
    # plt.colorbar(image, ax=ax1)
    
    # Thresholded mask on right
    mask = ax2.imshow((volume_seq[0, 0] >= min_thresh) & (volume_seq[0, 0] <= max_thresh), cmap='binary')
    ax2.set_title(f"{title} - Threshold Mask")
    # Add legend (black = 0, white = 1)
    # ax2.legend(['0', '1'], loc='upper right')

    # Create sliders
    ax_slice = plt.axes([0.25, 0.25, 0.5, 0.03])
    ax_time = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_min_thresh = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_max_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])

    data_min, data_max = np.min(volume_seq), np.max(volume_seq)
    
    time_slider = Slider(ax_time, 'Time', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice, 'Slice', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    min_thresh_slider = Slider(ax_min_thresh, 'Min Threshold', data_min, data_max, valinit=min_thresh)
    max_thresh_slider = Slider(ax_max_thresh, 'Max Threshold', data_min, data_max, valinit=max_thresh)
    
    scrolling_slider = [time_slider]

    def update(val, scrolling_slider=None, slider=None):
        current_slice = volume_seq[int(time_slider.val), int(slice_slider.val)]
        if apply_window:
            image.set_data(np.clip(current_slice, a_min=-40, a_max=120))
        else:
            image.set_data(current_slice)
        # Update threshold mask
        thresh_mask = (current_slice >= min_thresh_slider.val) & (current_slice <= max_thresh_slider.val)
        mask.set_data(thresh_mask)
        
        
        fig.canvas.draw_idle()
        if scrolling_slider is not None and slider is not None:
            scrolling_slider[0] = slider

    # Connect update function to all sliders
    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))
    min_thresh_slider.on_changed(update)
    max_thresh_slider.on_changed(update)

    def on_scroll(event):
        if event.button == 'up':
            scrolling_slider[0].set_val(min(scrolling_slider[0].val + scrolling_slider[0].valstep, 
                                          scrolling_slider[0].valmax))
        elif event.button == 'down':
            scrolling_slider[0].set_val(max(scrolling_slider[0].val - scrolling_slider[0].valstep, 
                                          scrolling_slider[0].valmin))
        
        fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes in [ax1, ax2]:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, (ax1, ax2)

def interactive_plot_with_mask(volume_seq, title="", cmap='grey', show=True, initial_min_objects=500, threshold_min=-25, threshold_max=150, apply_window=True):
    """Like interactive_plot but shows the mask from get_mask with adjustable min_objects parameter"""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    # plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.9, wspace=0.0, hspace=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    # Original image on left
    if apply_window:
        image = ax1.imshow(np.clip(volume_seq[0, 0], a_min=-40, a_max=120), cmap=cmap)
    else:
        image = ax1.imshow(volume_seq[0, 0], cmap=cmap)
    ax1.set_title(f"{title} - Original")
    
    # Mask in middle
    mask_img = get_2d_mask(volume_seq[0, 0], 
                        threshold_min=threshold_min, 
                        threshold_max=threshold_max, 
                        remove_small_objects_size=initial_min_objects,
                        structuring_element_dims=(1, 7)
                        )
    
    binary_cmap = ListedColormap(['black', 'white'])
    mask_plot = ax2.imshow(mask_img, cmap=binary_cmap)
    ax2.set_title(f"{title} - Brain Mask")
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=binary_cmap(0), label='0 - Non-Brain'),
        plt.Rectangle((0,0), 1, 1, facecolor=binary_cmap(1), label='1 - Brain', )
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0),
              loc='upper right', ncol=1)
    
    # Masked image on right
    masked_img = volume_seq[0, 0] * mask_img
    if apply_window:
        masked_image = ax3.imshow(np.clip(masked_img, a_min=-40, a_max=120), cmap=cmap)
    else:
        masked_image = ax3.imshow(masked_img, cmap=cmap)
    ax3.set_title(f"{title} - Masked Brain")

    # Create sliders
    ax_slice = plt.axes([0.15, 0.15, 0.32, 0.03])
    ax_time = plt.axes([0.53, 0.15, 0.32, 0.03])
    ax_min_objects = plt.axes([0.15, 0.13, 0.7, 0.03])
    ax_thresh_min = plt.axes([0.15, 0.11, 0.32, 0.03])  # Left half
    ax_thresh_max = plt.axes([0.53, 0.11, 0.32, 0.03])  # Right half
    ax_struct_rows = plt.axes([0.15, 0.09, 0.32, 0.03])  # Left half
    ax_struct_cols = plt.axes([0.53, 0.09, 0.32, 0.03])  # Right half
    
    slice_slider = Slider(ax_slice, 'Slice/Time', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    time_slider = Slider(ax_time, '', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    min_objects_slider = Slider(ax_min_objects, 'Min Objects Size', 0, 2000, valinit=initial_min_objects)
    thresh_min_slider = Slider(ax_thresh_min, 'Threshold Min/Max', -100, 200, valinit=threshold_min, valstep=1)
    thresh_max_slider = Slider(ax_thresh_max, '', -100, 400, valinit=threshold_max, valstep=1)
    struct_rows_slider = Slider(ax_struct_rows, 'Struct Rows/Cols', 1, 30, valinit=1, valstep=1)
    struct_cols_slider = Slider(ax_struct_cols, '', 1, 30, valinit=7, valstep=1)

    def update(val):
        current_slice = volume_seq[int(time_slider.val), int(slice_slider.val)]
        if apply_window:
            image.set_data(np.clip(current_slice, a_min=-40, a_max=120))
        else:
            image.set_data(current_slice)
            
        # Update mask with threshold parameters
        mask_img = get_2d_mask(current_slice, 
                          threshold_min=thresh_min_slider.val,
                          threshold_max=thresh_max_slider.val,
                          remove_small_objects_size=int(min_objects_slider.val),
                          structuring_element_dims=(int(struct_rows_slider.val), int(struct_cols_slider.val)))
        mask_plot.set_data(mask_img)
        
        # Update masked image
        masked_img = current_slice * mask_img
        if apply_window:
            masked_image.set_data(np.clip(masked_img, a_min=-40, a_max=120))
        else:
            masked_image.set_data(masked_img)
        
        fig.canvas.draw_idle()

    # Connect update function to all sliders
    time_slider.on_changed(update)
    slice_slider.on_changed(update)
    min_objects_slider.on_changed(update)
    thresh_min_slider.on_changed(update)
    thresh_max_slider.on_changed(update)
    struct_rows_slider.on_changed(update)
    struct_cols_slider.on_changed(update)

    def on_scroll(event):
        if event.inaxes == ax_time:
            slider = time_slider
        elif event.inaxes == ax_min_objects:
            slider = min_objects_slider
        elif event.inaxes == ax_thresh_min:
            slider = thresh_min_slider
        elif event.inaxes == ax_thresh_max:
            slider = thresh_max_slider
        elif event.inaxes == ax_struct_rows:
            slider = struct_rows_slider
        elif event.inaxes == ax_struct_cols:
            slider = struct_cols_slider
        else:
            slider = slice_slider
            
        if event.button == 'up':
            slider.set_val(min(slider.val + slider.valstep, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - slider.valstep, slider.valmin))

    def on_motion(event):
        if event.inaxes in [ax1, ax2, ax3]:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, (ax1, ax2, ax3)

def interactive_plot_with_3d_mask(volume_seq, title="", show=True, 
                                 initial_min_objects=500, 
                                 threshold_min=-25, threshold_max=150,
                                 initial_morph_y=3, initial_morph_x=3, initial_morph_z=3,
                                 initial_conn_y=3, initial_conn_x=3, initial_conn_z=3,
                                 apply_window=True,
                                 cmap='grey'):
    """Like interactive_plot_with_mask but uses get_3d_mask with adjustable 3D morphology and connectivity parameters"""
    plt.ion()  # Turn on interactive mode
    
    # Create figure with space for colorbars below
    fig = plt.figure(figsize=(24, 8))
    gs = plt.GridSpec(2, 3, height_ratios=[20, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    cax1 = plt.subplot(gs[1, 0])  # colorbar for original image
    cax3 = plt.subplot(gs[1, 2])  # colorbar for masked image
    
    plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.9, wspace=0.0, hspace=0.0)
    # plt.subplots_adjust(bottom=0.35, wspace=0.0, hspace=0.0)
    # plt.subplots_adjust(wspace=0, hspace=0)
    
    # Original image on left
    if apply_window:
        image = ax1.imshow(np.clip(volume_seq[0, 0], a_min=-40, a_max=120), cmap=cmap)
    else:
        image = ax1.imshow(volume_seq[0, 0], cmap=cmap)
    # Add colorbar below image
    plt.colorbar(image, cax=cax1, orientation='horizontal')
    ax1.set_title(f"{title} - Original")
    
    # Initial 3D mask
    volume_masks = [get_3d_mask(volume_seq[t], 
                           threshold_min=threshold_min,
                           threshold_max=threshold_max,
                           remove_small_objects_size=initial_min_objects,
                           morphology_shape_2d=(1, 7),
                           morphology_shape_3d=(initial_morph_y, initial_morph_x, initial_morph_z),
                           connectivity_shape_3d=(initial_conn_y, initial_conn_x, initial_conn_z))
                        for t in range(volume_seq.shape[0])]
    
    binary_cmap = ListedColormap(['black', 'white'])
    # Mask in middle with binary_r colormap
    mask_plot = ax2.imshow(volume_masks[0][0], cmap=binary_cmap)
    ax2.set_title(f"{title} - Brain Mask")
    
    # Add legend for mask using the same colormap
    
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=binary_cmap(0), label='0 - Non-Brain'),
        plt.Rectangle((0,0), 1, 1, facecolor=binary_cmap(1), label='1 - Brain', )
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0),
              loc='upper right', ncol=1)

    # Masked image on right
    masked_img = apply_mask(volume_seq[0, 0], volume_masks[0][0])
    if apply_window:
        masked_image = ax3.imshow(np.clip(masked_img, a_min=-40, a_max=120), cmap=cmap)
    else:
        masked_image = ax3.imshow(masked_img, cmap=cmap)
    # Add colorbar below masked image
    plt.colorbar(masked_image, cax=cax3, orientation='horizontal')
    ax3.set_title(f"{title} - Masked Brain")

    # Create sliders
    ax_slice = plt.axes([0.15, 0.25, 0.32, 0.03])
    ax_time = plt.axes([0.53, 0.25, 0.32, 0.03])
    ax_min_objects = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_thresh_min = plt.axes([0.15, 0.17, 0.32, 0.03])
    ax_thresh_max = plt.axes([0.53, 0.17, 0.32, 0.03])
    
    # 3D morphology sliders (y, x, z order)
    ax_morph_y = plt.axes([0.15, 0.14, 0.22, 0.03])
    ax_morph_x = plt.axes([0.39, 0.14, 0.22, 0.03])
    ax_morph_z = plt.axes([0.63, 0.14, 0.22, 0.03])
    
    # 3D connectivity sliders (y, x, z order)
    # ax_conn_y = plt.axes([0.15, 0.07, 0.22, 0.03])
    # ax_conn_x = plt.axes([0.39, 0.07, 0.22, 0.03])
    # ax_conn_z = plt.axes([0.63, 0.07, 0.22, 0.03])
    
    slice_slider = Slider(ax_slice, 'Slice/Time', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    time_slider = Slider(ax_time, '', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    min_objects_slider = Slider(ax_min_objects, 'Min Objects Size', 0, 2000, valinit=initial_min_objects, valstep=10)
    thresh_min_slider = Slider(ax_thresh_min, 'Threshold Min/Max', -100, 200, valinit=threshold_min, valstep=1)
    thresh_max_slider = Slider(ax_thresh_max, '', -100, 400, valinit=threshold_max, valstep=1)
    
    morph_y_slider = Slider(ax_morph_y, 'Morph Y/X/Z', 1, 10, valinit=initial_morph_y, valstep=2)
    morph_x_slider = Slider(ax_morph_x, '', 1, 10, valinit=initial_morph_x, valstep=2)
    morph_z_slider = Slider(ax_morph_z, '', 1, 10, valinit=initial_morph_z, valstep=2)
    
    # conn_y_slider = Slider(ax_conn_y, 'Connect Y/X/Z', 1, 10, valinit=initial_conn_y, valstep=2)
    # conn_x_slider = Slider(ax_conn_x, '', 1, 10, valinit=initial_conn_x, valstep=2)
    # conn_z_slider = Slider(ax_conn_z, '', 1, 10, valinit=initial_conn_z, valstep=2)

    def update_slice_time(val):
        current_volume = volume_seq[int(time_slider.val)]
        current_mask = volume_masks[int(time_slider.val)]
        current_slice_idx = int(slice_slider.val)
        
        if apply_window:
            image.set_data(np.clip(current_volume[current_slice_idx], a_min=-40, a_max=120))
        else:
            image.set_data(current_volume[current_slice_idx])
        # Update mask
        mask_plot.set_data(current_mask[current_slice_idx])
        # Update masked image
        masked_img = apply_mask(current_volume[current_slice_idx], current_mask[current_slice_idx])
        ic(masked_img.max(), masked_img.min())
        if apply_window:
            masked_image.set_data(np.clip(masked_img, a_min=-40, a_max=120))
        else:
            masked_image.set_data(masked_img)
        
        fig.canvas.draw_idle()

    def update_mask(val):
                    
        # Update 3D mask with all parameters
        volume_masks = [get_3d_mask(volume_seq[t],
                               threshold_min=thresh_min_slider.val,
                               threshold_max=thresh_max_slider.val,
                               remove_small_objects_size=int(min_objects_slider.val),
                               morphology_shape_2d=(3, 3),
                               morphology_shape_3d=(int(morph_y_slider.val),
                                                  int(morph_x_slider.val),
                                                  int(morph_z_slider.val)))
                        for t in range(volume_seq.shape[0])]
        current_mask = volume_masks[int(time_slider.val)]
        # Update masked image
        masked_img = apply_mask(volume_seq[int(time_slider.val)][int(slice_slider.val)], current_mask[int(slice_slider.val)])
        if apply_window:
            masked_image.set_data(np.clip(masked_img, a_min=-40, a_max=120))
        else:
            masked_image.set_data(masked_img)
        mask_plot.set_data(current_mask[int(slice_slider.val)])
    # Connect update function to all sliders
    time_slider.on_changed(update_slice_time)
    slice_slider.on_changed(update_slice_time)
    min_objects_slider.on_changed(update_mask)
    thresh_min_slider.on_changed(update_mask)
    thresh_max_slider.on_changed(update_mask)
    morph_y_slider.on_changed(update_mask)
    morph_x_slider.on_changed(update_mask)
    morph_z_slider.on_changed(update_mask)
    # conn_y_slider.on_changed(update_mask)
    # conn_x_slider.on_changed(update_mask)
    # conn_z_slider.on_changed(update_mask)

    def on_scroll(event):
        if event.inaxes == ax_time:
            slider = time_slider
        elif event.inaxes == ax_min_objects:
            slider = min_objects_slider
        elif event.inaxes == ax_thresh_min:
            slider = thresh_min_slider
        elif event.inaxes == ax_thresh_max:
            slider = thresh_max_slider
        elif event.inaxes == ax_morph_y:
            slider = morph_y_slider
        elif event.inaxes == ax_morph_x:
            slider = morph_x_slider
        elif event.inaxes == ax_morph_z:
            slider = morph_z_slider
        # elif event.inaxes == ax_conn_y:
        #     slider = conn_y_slider
        # elif event.inaxes == ax_conn_x:
        #     slider = conn_x_slider
        # elif event.inaxes == ax_conn_z:
        #     slider = conn_z_slider
        else:
            slider = slice_slider
            
        if event.button == 'up':
            slider.set_val(min(slider.val + slider.valstep, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - slider.valstep, slider.valmin))

    def on_motion(event):
        if event.inaxes in [ax1, ax2, ax3]:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, (ax1, ax2, ax3)

def interactive_plot_with_binary_mask(volume_seq, volume_mask, title="", show=True):
    """
    Displays an interactive plot of a volume with a binary mask overlay.
    
    Parameters:
    - volume_seq: 4D numpy array with shape (time, slices, height, width)
    - volume_mask: 3D numpy array containing binary masks
    - title: Title of the plot (default: "")
    - show: Whether to display the plot immediately (default: True)
    
    Returns:
    - fig: Matplotlib figure object
    - ax: Matplotlib axes object
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Display the initial volume slice
    image = ax.imshow(volume_seq[0, 0], cmap='gray')
    
    # Overlay the initial mask with transparency in grayscale
    mask_overlay = ax.imshow(volume_mask[0], cmap='grey', alpha=0.5)
    
    ax.set_title(title)
    
    # Create sliders for slice and time
    ax_slice = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])
    
    time_slider = Slider(ax_time, 'Time', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice, 'Slice', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    
    scrolling_slider = [time_slider]
    
    plt.colorbar(image, ax=ax)
    
    def update(val, scrolling_slider=None, slider=None):
        current_time = int(time_slider.val)
        current_slice = int(slice_slider.val)
        
        # Update the volume image
        image.set_data(volume_seq[current_time, current_slice])
        
        # Update the mask overlay
        mask_overlay.set_data(volume_mask[current_slice])
        
        fig.canvas.draw_idle()
        
        if scrolling_slider is not None and slider is not None:
            scrolling_slider[0] = slider
    
    # Connect the update function to the sliders
    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))
    
    def on_scroll(event):
        if event.button == 'up':
            scrolling_slider[0].set_val(min(scrolling_slider[0].val + scrolling_slider[0].valstep, scrolling_slider[0].valmax))
        elif event.button == 'down':
            scrolling_slider[0].set_val(max(scrolling_slider[0].val - scrolling_slider[0].valstep, scrolling_slider[0].valmin))
    
    def on_motion(event):
        if event.inaxes == ax:
            plt.gcf().canvas.set_cursor(1)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    if show:
        plt.show(block=True)
    
    return fig, ax

def save_slices_with_mask(volume_seq, folder_path):
    volume_mask = np.array([get_2d_mask(s) for s in volume_seq[0]])
    volume_mask = get_largest_connected_component(volume_mask)
    # interactive_plot_with_binary_mask(volume_seq, volume_mask, title=folder_path.split('/')[-1])
    if not os.path.exists(f"Images"):
        os.makedirs(f"Images")
    for i, slice in enumerate(volume_seq[0]):
        fig, ax = plt.subplots()
        
        # Display the initial volume slice
        image = ax.imshow(slice, cmap='grey')
        
        # Overlay the initial mask with transparency in grayscale
        mask_overlay = ax.imshow(volume_mask[i], cmap='gray', alpha=0.2)
        
        ax.set_title(f"{folder_path.split('/')[-1]} - Slice {i}")
        if not os.path.exists(f"Images/{folder_path.split('/')[-1]}"):
            os.makedirs(f"Images/{folder_path.split('/')[-1]}")
        # Save image
        plt.savefig(f"Images/{folder_path.split('/')[-1]}/slice_{i}.jpg")
        plt.close()


def interactive_plot_with_bilateral_filter(volume_seq, title="", cmap='grey', show=True, initial_sigma_space=3.0, initial_sigma_intensity=15, windowing_params=(80, 160)):
    """
    Interactive plot for a volume sequence with adjustable sigma_space and sigma_intensity for bilateral filtering.
    
    Args:
        volume_seq: 4D numpy array (time, slices, height, width)
        title: Title of the plot
        show: Whether to display the plot immediately
        initial_sigma_space: Initial value for sigma_space slider
        initial_sigma_intensity: Initial value for sigma_intensity slider
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9, wspace=0.3)

    if windowing_params:
        try:
            volume_seq = apply_window(volume_seq, *windowing_params)
        except Exception as e:
            print(f"Error applying windowing: {e}")

    # Original image on the left
    image_original = ax1.imshow(volume_seq[0, 0], cmap=cmap)
    ax1.set_title(f"{title} - Original")
    plt.colorbar(image_original, ax=ax1)
    ax1.axis('off')
    
    # Custom filtered image on the right
    filtered_image_data = apply_bilateral_filter(volume_seq[0, 0], sigma_space=initial_sigma_space, sigma_intensity=initial_sigma_intensity)
    image_filtered = ax2.imshow(filtered_image_data, cmap=cmap)
    ax2.set_title(f"{title} - Bilateral Filter")
    plt.colorbar(image_filtered, ax=ax2)
    ax2.axis('off')
    
    # Create sliders
    ax_slice_slider = plt.axes([0.15, 0.25, 0.7, 0.03])
    ax_time_slider = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_sigma_space_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_sigma_intensity_slider = plt.axes([0.15, 0.10, 0.7, 0.03])
    
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    time_slider = Slider(ax_time_slider, 'Time', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    sigma_space_slider = Slider(ax_sigma_space_slider, 'Sigma Space', 0, 20.0, valinit=initial_sigma_space, valstep=1)
    sigma_intensity_slider = Slider(ax_sigma_intensity_slider, 'Sigma Intensity', 0, 30.0, valinit=initial_sigma_intensity, valstep=0.5)
    
    def update(val):
        current_time = int(time_slider.val)
        current_slice = int(slice_slider.val)
        sigma_space = sigma_space_slider.val
        sigma_intensity = sigma_intensity_slider.val
        
        original_data = volume_seq[current_time, current_slice]
        filtered_data = apply_bilateral_filter(original_data, sigma_space, sigma_intensity)
        
        image_original.set_data(original_data)
        image_filtered.set_data(filtered_data)

        fig.canvas.draw_idle()
    
    # Connect update function to sliders
    time_slider.on_changed(update)
    slice_slider.on_changed(update)
    sigma_space_slider.on_changed(update)
    sigma_intensity_slider.on_changed(update)
    
    def on_scroll(event):
        if event.inaxes == ax_time_slider:
            time_slider.set_val(max(min(time_slider.val + event.step, time_slider.valmax), time_slider.valmin))
        elif event.inaxes == ax_slice_slider:
            slice_slider.set_val(max(min(slice_slider.val + event.step, slice_slider.valmax), slice_slider.valmin))
        elif event.inaxes == ax_sigma_space_slider:
            sigma_space_slider.set_val(min(sigma_space_slider.val + 0.1, sigma_space_slider.valmax))
        elif event.inaxes == ax_sigma_intensity_slider:
            sigma_intensity_slider.set_val(min(sigma_intensity_slider.val + 0.1, sigma_intensity_slider.valmax))
        
    def on_motion(event):
        if event.inaxes in [ax1, ax2]:
            plt.gcf().canvas.set_cursor(1)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    if show:
        plt.show(block=True)
    
    return fig, (ax1, ax2)


def overlay_volumes(base_volume, overlay_volume, title=None, show=True):
    """
    Display one 3D volume overlaid on another with different colormaps and an interactive slice slider.
    
    Args:
        base_volume: The base volume to display in grayscale (3D array)
        overlay_volume: The volume to overlay in hot colormap with transparency (3D array)
        title: Optional title for the plot
        show: Whether to display the plot immediately
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if title:
        ax.set_title(title)
    
    # Display initial slice
    base_img = ax.imshow(base_volume[0], cmap='gray')
    overlay_img = ax.imshow(overlay_volume[0], cmap='hot', alpha=0.5)
    plt.colorbar(overlay_img, ax=ax)
    ax.axis('off')
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(base_volume)-1, valinit=0, valstep=1)
    
    def update(val):
        slice_idx = int(slider.val)
        base_img.set_data(base_volume[slice_idx])
        overlay_img.set_data(overlay_volume[slice_idx])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add scroll functionality
    def on_scroll(event):
        if event.button == 'up':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - 1, slider.valmin))
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    if show:
        plt.show()
    
    return fig, ax

def overlay_volume_sequence(volume_seq, title=None, show=True):
    """
    Display each volume in a sequence overlaid on the first volume with different colormaps 
    and interactive slice sliders.
    
    Args:
        volume_seq: 4D array of shape (T, Y, Z, X) representing the volume sequence
        title: Optional title for the plot
        show: Whether to display the plot immediately
    """
    n_volumes = volume_seq.shape[0]
    base_volume = volume_seq[0]
    
    # Create subplot for each volume after the first
    fig, axes = plt.subplots(1, n_volumes-1, figsize=(5*n_volumes, 8))
    if n_volumes == 2:  # Handle case of single subplot
        axes = [axes]
        
    if title:
        fig.suptitle(title)
        
    # Display initial slices
    base_imgs = []
    overlay_imgs = []
    for i, ax in enumerate(axes):
        base_img = ax.imshow(base_volume[0], cmap='gray')
        overlay_img = ax.imshow(volume_seq[i+1][0], cmap='hot', alpha=0.5)
        plt.colorbar(overlay_img, ax=ax)
        ax.axis('off')
        ax.set_title(f'Volume {i+1}')
        base_imgs.append(base_img)
        overlay_imgs.append(overlay_img)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(base_volume)-1, valinit=0, valstep=1)
    
    def update(val):
        slice_idx = int(slider.val)
        for i in range(len(axes)):
            base_imgs[i].set_data(base_volume[slice_idx])
            overlay_imgs[i].set_data(volume_seq[i+1][slice_idx])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add scroll functionality
    def on_scroll(event):
        if event.button == 'up':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - 1, slider.valmin))
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


from matplotlib import animation, rc

def create_animation(ims):
    rc('animation', html='jshtml')
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

def downsample_volume(volume, factor):
    """Downsamples the volume by the given factor."""
    return scipy.ndimage.zoom(volume, (1/factor, 1/factor, 1/factor), order=1)


def plot_3d_skeleton(ctvol, outprefix='./'):
    """
    Make a 3D plot of the skeleton.
    <units> either 'HU' or 'normalized' determines the thresholds
    """
    #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = ctvol.transpose(2,1,0)
    p = np.flip(p, axis = 0) #need this line or else the patient is upside-down
    
    #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, _ignore1, _ignore2 = measure.marching_cubes(p)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.savefig(outprefix+'_3D_Bones.png')
    plt.close()    

def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].
    
    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    #First fix the grayscale colors.
    #imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    #If you do not truncate to a 256 range, imageio will do so on a per-slice
    #basis, which creates weird brightening and darkening artefacts in the gif.
    #Thus, before making the gif, you should truncate to the 0-256 range
    #and cast to a uint8 (the dtype imageio requires):
    #how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (  ((ctvol+800)*(255))/(400+800)  ).astype('uint8')
    
    #Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[0]):
            images.append(ctvol[slicenum,:,:])
        imageio.mimsave(outprefix+'_axial.gif',images)
        print('\t\tdone with axial gif')
    
    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:,slicenum,:])
        imageio.mimsave(outprefix+'_coronal.gif',images)
        print('\t\tdone with coronal gif')
    
    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:,:,slicenum])
        imageio.mimsave(outprefix+'_sagittal.gif',images)
        print('\t\tdone with sagittal gif')

def multi_folder_cycle_iplot_all(folder_list, windowing_params=None, nrows=None, cmap='grey'):
    """
    Render interactive cycling of volume sequences from several folders in a single window
    with different subplots (one per folder). The folders are expected to contain
    .npy files. Each .npy file is assumed to be a 4D numpy array of shape 
    (time, slice, height, width). 

    There are two unified sliders below the subplots controlling the displayed time and slice
    indices. Keyboard events act on ALL subplots simultaneously:
      - Pressing Enter cycles forward the volume for every folder (if available).
      - Pressing Backspace cycles backward for every folder (if available).
      
    Parameters:
      folder_list : list[str]
          List of folder paths (each containing .npy volume files).
      windowing_params : tuple, optional
          If provided, applied to the volume slices via apply_window().
      nrows : int, optional
          Number of subplot rows (if not provided, determined automatically).
      
    Returns:
      fig, axes, folder_states : tuple
          fig       : the matplotlib figure.
          axes      : the array of Axes.
          folder_states: a list of dictionaries, one per folder. Each dict includes:
                           "folder"         : folder path as string.
                           "files"          : list of npy file names in the folder.
                           "current_idx"    : current index into the files list.
                           "current_volume" : current 4D volume (np.load result).
                           "current_title"  : current file name (without extension).
                           "ax"             : the Axes for displaying.
                           "image_handle"   : the Image returned by imshow.
    """
    # Build a state dict for each folder.
    folder_states = []
    for folder in folder_list:
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        if not files:
            print(f"No npy files found in folder: {folder}")
            continue
        # Load the first volume in the folder.
        filepath = os.path.join(folder, files[0])
        volume = np.load(filepath)
        title = os.path.splitext(files[0])[0]
        folder_states.append({
            "folder": folder,
            "files": files,
            "current_idx": 0,
            "current_volume": volume,
            "current_title": title
        })

    if not folder_states:
        print("No folders with npy volumes found.")
        return None, None, None

    num_folders = len(folder_states)
    if nrows is None:
        nrows = int(np.sqrt(num_folders))
        if nrows == 0:
            nrows = 1
    ncols = (num_folders + nrows - 1) // nrows

    # Create the overall figure and an array of axes.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
    axes_list = axes.flatten()

    # For each folder, show the initial slice (time=0, slice=0)
    for i, state in enumerate(folder_states):
        ax = axes_list[i]
        if windowing_params is not None:
            img_data = apply_window(state["current_volume"][0, 0], *windowing_params)
        else:
            img_data = state["current_volume"][0, 0]
        image = ax.imshow(img_data, cmap=cmap)
        ax.set_title(f"{state['current_title']} ({state['folder']})")
        plt.colorbar(image, ax=ax)
        state["ax"] = ax
        state["image_handle"] = image

    # Hide any extra axes.
    for j in range(num_folders, len(axes_list)):
        axes_list[j].axis('off')

    # Create two global sliders for time and slice.
    slider_ax_width = 0.5
    slider_ax_height = 0.03
    slider_left = 0.25
    ax_slice_slider = fig.add_axes([slider_left, 0.1, slider_ax_width, slider_ax_height])
    ax_time_slider  = fig.add_axes([slider_left, 0.05, slider_ax_width, slider_ax_height])
    max_time = max(state["current_volume"].shape[0] for state in folder_states) - 1
    max_slice = max(state["current_volume"].shape[1] for state in folder_states) - 1
    time_slider = Slider(ax_time_slider, 'Time', 0, max_time, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, max_slice, valinit=0, valstep=1)

    def update_all_images(val):
        t_val = int(time_slider.val)
        s_val = int(slice_slider.val)
        for state in folder_states:
            vol = state["current_volume"]
            # Clamp slider values to the current volume's dimensions.
            t = min(t_val, vol.shape[0]-1)
            s = min(s_val, vol.shape[1]-1)
            if windowing_params is not None:
                new_data = apply_window(vol[t, s], *windowing_params)
            else:
                new_data = vol[t, s]
            state["image_handle"].set_data(new_data)
        fig.canvas.draw_idle()

    time_slider.on_changed(update_all_images)
    slice_slider.on_changed(update_all_images)

    def on_scroll(event):
        if event.inaxes == ax_time_slider:
            new_val = min(time_slider.val + event.step, time_slider.valmax)
            time_slider.set_val(new_val)
        elif event.inaxes == ax_slice_slider:
            new_val = min(slice_slider.val + event.step, slice_slider.valmax)
            slice_slider.set_val(new_val)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key_press(event):
        if event.key == 'enter':
            # Cycle forward for every folder state.
            for state in folder_states:
                if state["current_idx"] < len(state["files"]) - 1:
                    state["current_idx"] += 1
                    filepath = os.path.join(state["folder"], state["files"][state["current_idx"]])
                    new_vol = np.load(filepath)
                    state["current_volume"] = new_vol
                    state["current_title"] = os.path.splitext(state["files"][state["current_idx"]])[0]
                    state["ax"].set_title(f"{state['current_title']} ({state['folder']})")
                else:
                    print(f"Folder '{state['folder']}': no more volumes to display.")
            # Update the global slider maximums
            new_max_time = max(s["current_volume"].shape[0] for s in folder_states) - 1
            new_max_slice = max(s["current_volume"].shape[1] for s in folder_states) - 1
            time_slider.valmax = new_max_time
            time_slider.ax.set_xlim(time_slider.valmin, time_slider.valmax)
            slice_slider.valmax = new_max_slice
            slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
            update_all_images(None)
        elif event.key == 'backspace':
            # Cycle backward for every folder state.
            for state in folder_states:
                if state["current_idx"] > 0:
                    state["current_idx"] -= 1
                    filepath = os.path.join(state["folder"], state["files"][state["current_idx"]])
                    new_vol = np.load(filepath)
                    state["current_volume"] = new_vol
                    state["current_title"] = os.path.splitext(state["files"][state["current_idx"]])[0]
                    state["ax"].set_title(f"{state['current_title']} ({state['folder']})")
                else:
                    print(f"Folder '{state['folder']}': already at the first volume.")
            new_max_time = max(s["current_volume"].shape[0] for s in folder_states) - 1
            new_max_slice = max(s["current_volume"].shape[1] for s in folder_states) - 1
            time_slider.valmax = new_max_time
            time_slider.ax.set_xlim(time_slider.valmin, time_slider.valmax)
            slice_slider.valmax = new_max_slice
            slice_slider.ax.set_xlim(slice_slider.valmin, slice_slider.valmax)
            update_all_images(None)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show(block=True)
    return fig, axes, folder_states

