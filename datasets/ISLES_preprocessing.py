from preprocessing import *
from visualization import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json

# ------------------ Nifti Scan Functions ------------------
def load_nii_scan(file_path: str) -> np.ndarray:
    nii = nib.load(file_path)
    scan = nii.get_fdata()
    scan = scan.transpose((3, 2, 0, 1))
    scan = np.rot90(scan, k=1, axes=(2, 3))
    return scan

def save_nii_scan(volume: np.ndarray, file_path: str):
    nii = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nii, file_path)

# ----------------------------------------------------------

def preprocess_scan(volume_seq: np.ndarray, 
               window_params: tuple|str|None=(200, 400), 
               filter=True,
               extract_brain=True,
               standardize=True,
               correct_motion=True,
               reference_index=3,
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
        volume_mask = get_3d_mask(volume_seq[reference_index], threshold_min=-10, threshold_max=120,morphology_shape_3d=(3, 3, 3), morphology_shape_2d=(3, 5), adaptive=False)

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
    
    save_path = f'Experiments/ISLES_PreprocessedData'

        
    with open('selected_paths.json', 'r') as f:
        paths = json.load(f)
    for path in paths[3:]:
        # v = load_nii_scan(path)
        nii = nib.load(path)
        hdr = nii.header

        # if hdr.get('dim')[3] != 22:
        #     continue

        scan = nii.get_fdata().astype('float32')
        scan = scan.transpose((3, 2, 0, 1))

        v = np.rot90(scan, k=1, axes=(2, 3))
        T = v.shape[0]

        # Uniformly (integer precisions) subsample along time axis to get 18 volumes
        # indices = np.linspace(0, T - 1, 18, dtype=int)
        
        # Acutally uniformly subsample along time axis to get 18 volumes
        step = T // 18
        t = step * 18
        pad = (T - t) // 2
        # print(f"T: {T}, step: {step}, t: {t}, pad: {pad}")
        indices = np.arange(1, t + 1, step, dtype=int)
        # print(f"{indices=}")

        preprocessed = preprocess_scan(v[indices], verbose=False)

        # multiple samples of 16 slices from scans with more than 16 slices
        if v.shape[1] == 22:
            for i in range(2, v.shape[1]-16):
                sub = preprocessed[:, i:i+16, :, :]
                # interactive_plot(sub, windowing_params=(40, 80))
                save_volume(sub, os.path.join(save_path, path.split('/')[-3] + f'_{i}.npy'))
        else:
            save_volume(preprocessed, os.path.join(save_path, path.split('/')[-3] + '.npy'))
            # interactive_plot_with_threshold(v) # --> The value range is not the typical HU range. Instead [-23, 1200+]
            # Calculate step size to get exactly 18 volumes
            # Sample indices to get exactly 18 volumes
            # interactive_plot_with_3d_mask(v[:1], threshold_min=-21)
            # interactive_plot(preprocessed, windowing_params=(40, 80))

    # for path in os.listdir(save_path):
    #     v = np.load(os.path.join(save_path, path)).astype('float32')
    #     print(v.dtype)
    #     np.save(os.path.join(save_path, path), v)