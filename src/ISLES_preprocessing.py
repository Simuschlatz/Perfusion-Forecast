from preprocessing import *
from visualization import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json

def preprocess_scan(volume_seq: np.ndarray, 
               window_params: tuple|str|None=(200, 400), 
               filter=True,
               extract_brain=True,
               standardize=True,
               correct_motion=True,
               reference_index=0,
               slice_based=False,
               verbose=True) -> np.ndarray:
    """
    Processes the DICOM files in a folder with folder path `folder_path`.
    Each folder contains the entire perfusion volume sequence as DICOM datasets
    The objective is to convert the sequence into a 4D array of CT volumes that are
    in HU, windowed, brain-extracted, registered, filtered, standardized

    Parameters:
    - `window_params` (tuple | str | None, optional): Controls windowing. If None, no windowing is applied. If tuple, it is interpreted as (window_center, window_width). 
    If str, it is interpreted as a windowing type. Defaults to (80, 160).
    - `extract_brain` (bool, optional): Whether to extract the brain from the volume sequence. Defaults to True.
    - `correct_motion` (bool, optional): Whether to correct motion in the volume sequence. Defaults to True.
    - `reference_index` (int, optional): Index of the reference volume in the sequence to which other volumes will be registered. Defaults to 1.
    - `spatial_downsampling_factor` (int, optional): Factor by which to downsample the volume sequence in the spatial dimensions. Defaults to 4.
    - `temporal_downsampling_factor` (int, optional): Factor by which to downsample the volume sequence in the temporal dimension. Defaults to 1.
    """

    if extract_brain: # Calculate single brain mask for all volumes before windowing
        volume_mask = get_3d_mask(volume_seq[reference_index], threshold_min=-21, morphology_shape_3d=(3, 3, 5))

    if window_params is not None:
        if type(window_params) == str:
            window_center, window_width = get_window_from_type(window_params)
        else:
            window_center, window_width = window_params
        volume_seq = apply_window(volume_seq, window_center, window_width)
        if verbose: ic(volume_seq.max(), volume_seq.min(), volume_seq.dtype)

    if filter: 
        volume_seq = filter_volume_seq(volume_seq)
    
    if correct_motion:
        volume_seq = rigid_register_volume_sequence(volume_seq, reference_index=reference_index, verbose=verbose)

    if extract_brain: # Apply brain mask to all registered volumes
        volume_seq = apply_mask(volume_seq, volume_mask)

    if verbose: print(f"Done!")
    # Standardization
    # if standardize:
    #     if slice_based: # Standardize each slice sequence individually (for 2D video prediction)
    #         for slice_idx in range(volume_seq.shape[1]):
    #             volume_seq[:, slice_idx] = (volume_seq[:, slice_idx] - np.mean(volume_seq[:, slice_idx])) / np.std(volume_seq[:, slice_idx])
    #     else: # Standardize the entire volume sequence (for 3D video prediction)
    #         volume_seq = (volume_seq - np.mean(volume_seq)) / np.std(volume_seq)
    return volume_seq

if __name__ == "__main__":
    
    with open('selected_paths.json', 'r') as f:
        paths = json.load(f)
    for path in paths[3:]:
        v = load_nii_scan(path)
        # interactive_plot_with_threshold(v) # --> The value range is not the typical HU range. Instead [-23, 1200+]
        # interactive_plot_with_3d_mask(v[:1], threshold_min=-21)
        preprocessed = preprocess_scan(v[::10])
        interactive_plot(preprocessed, windowing_params=(40, 80))