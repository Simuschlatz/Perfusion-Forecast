if __name__ == "__main__":
    from preprocessing import get_volume, save_volume, rigid_register_volume_sequence, load_folder_paths
    from scipy import ndimage
    import os
    import SimpleITK as sitk
    import numpy as np

    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

    from skimage import morphology
    from scipy import ndimage
    from numba import jit

    @jit(nopython=True)
    def get_threshold_mask(slice, min_val=-40, max_val=150):
        return (min_val < slice) & (slice < max_val)

    @jit(nopython=True)
    def apply_mask(a: np.ndarray, mask: np.ndarray):
        min_val = np.min(a)
        return (a - min_val) * mask + min_val

    def mask_slice(slice, threshold_min=-40, threshold_max=150, structuring_element_dims=(3, 3)):
        smoothened = ndimage.gaussian_filter(slice, sigma=2.0)
        mask = get_threshold_mask(smoothened, min_val=threshold_min, max_val=threshold_max)
        mask = ndimage.binary_erosion(mask, np.ones(structuring_element_dims), iterations=2)
        # mask = ndimage.binary_erosion(mask, np.ones((3, 3)))
        # Remove small objects
        mask = morphology.remove_small_objects(mask, 800)
        # Fill small holes
        mask = ndimage.binary_dilation(mask, np.ones(structuring_element_dims), iterations=2)
        mask = ndimage.binary_fill_holes(mask)
        return apply_mask(slice, mask)

    def preprocess_slice_for_registration(slice, threshold_min=-40, threshold_max=150, structuring_element_dims=(3, 3), reg_window_min=20, reg_window_max=60):
        masked = mask_slice(slice, threshold_min=threshold_min, threshold_max=threshold_max, structuring_element_dims=structuring_element_dims)
        return np.clip(masked, reg_window_min, reg_window_max)

    def experiment_reg(moving_volume: np.ndarray, reference_volume: np.ndarray, n_samples: int = 5, 
                lr: float = 1.0, n_iters: int = 200, relaxation_factor: float = 0.99, multi_res: bool = False,
                gradient_magnitude_tolerance: float = 1e-5, max_step: float = 4.0, min_step: float = 5e-4, 
                spacing: tuple = (1, 1), smoothing_sigma: float = 2.0, interpolator=sitk.sitkLinear,
                mask=True, reg_window_min=-100, reg_window_max=300, threshold_min=0, threshold_max=500,
                shrink_factors: list = [4, 2, 1], smoothing_sigmas: list = [3, 2, 1],
                verbose: bool = False):
        
        Y, Z, X = moving_volume.shape
        
        if n_samples > Y: n_samples = Y
        elif n_samples < 1: n_samples = 1
        middle = Y // 2
        half_range = n_samples // 2
        slice_indices = np.linspace(middle - half_range, middle + half_range, n_samples, dtype=int)
        
        mean_metric_value = 0
        transforms = []
        weights = np.zeros(n_samples)
        
        # Register each sampled slice
        for pos, slice_idx in enumerate(slice_indices):
            
            # Standard preprocessing for registration
            if mask:
                moving_slice = preprocess_slice_for_registration(moving_volume[slice_idx], reg_window_min=reg_window_min, reg_window_max=reg_window_max, threshold_min=threshold_min, threshold_max=threshold_max)
                fixed_slice = preprocess_slice_for_registration(reference_volume[slice_idx], reg_window_min=reg_window_min, reg_window_max=reg_window_max, threshold_min=threshold_min, threshold_max=threshold_max)
            else:
                moving_slice = np.clip(moving_volume[slice_idx], 0, 160)
                fixed_slice = np.clip(reference_volume[slice_idx], 0, 160)
                

            # Set min pixel value to 0 so that background aligns with pixels moved in by transformation during registration
            min_pixel_value = np.min(moving_slice)
            moving_slice, fixed_slice = moving_slice - min_pixel_value, fixed_slice - min_pixel_value

            # Convert to SimpleITK images
            moving_image = sitk.GetImageFromArray(moving_slice)
            fixed_image = sitk.GetImageFromArray(fixed_slice)
            
            if smoothing_sigma and not multi_res:
                moving_image = sitk.DiscreteGaussian(moving_image, smoothing_sigma)
                fixed_image = sitk.DiscreteGaussian(fixed_image, smoothing_sigma)
            
            # Set 2D spacing
            moving_image.SetSpacing(spacing)
            fixed_image.SetSpacing(spacing)
            
            # Initialize 2D transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # Setup registration method
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMeanSquares()
            
            # Optimizer settings
            registration_method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=lr,
                maximumStepSizeInPhysicalUnits=max_step,
                minStep=min_step,
                numberOfIterations=n_iters,
                gradientMagnitudeTolerance=gradient_magnitude_tolerance,
                relaxationFactor=relaxation_factor,
                
            )
            if multi_res:
                registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
                registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
                registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(initial_transform)
            registration_method.SetInterpolator(interpolator)
    
            try:
                final_transform = registration_method.Execute(fixed_image, moving_image)
                params = final_transform.GetParameters()
                transforms.append(params)

                metric_value = registration_method.GetMetricValue()
                mean_metric_value += metric_value / n_samples
                position_weight = 1 - ((slice_idx - middle) / (middle-1)) # The further away from the center, the lower the weight
                weight = 1 / ((metric_value) + 1e-10) * position_weight ** 1.5
                weights[pos] = weight

            except RuntimeError as e:
                print(f"Registration failed for slice {slice_idx}: {e}")
                continue
        
        # Normalize weights
        weights = weights / np.sum(weights)
        # print(weights)
        # Compute weighted average transformation
        avg_angle = 0
        avg_tx = 0
        avg_ty = 0

        for params, weight in zip(transforms, weights):
            angle, tx, ty = params[0], params[1], params[2]
            avg_angle += angle * weight
            avg_tx += tx * weight
            avg_ty += ty * weight


            # Create final average transform
        final_transform = sitk.Euler2DTransform()
        final_transform.SetAngle(avg_angle)
        final_transform.SetTranslation((avg_tx, avg_ty))
        final_transform.SetCenter((127, 127))
        
        # Apply transform to each slice of moving volume
        registered_volume = np.zeros_like(moving_volume)
        for i in range(moving_volume.shape[0]):
            moving_slice = moving_volume[i]
            min_value = np.min(moving_slice)
            moving_slice = sitk.GetImageFromArray(moving_slice - min_value)
            registered_slice = sitk.Resample(
                moving_slice,
                final_transform,
                interpolator,
                0.0, # Min pixel value
                moving_slice.GetPixelID()
            )
            # Bring the slice back to its original value range by adding min_pixel_value
            registered_volume[i] = sitk.GetArrayFromImage(registered_slice) + min_value
        return registered_volume


    def mse(moving_volume: np.ndarray, reference_volume: np.ndarray):
        return np.mean(np.square(moving_volume - reference_volume))

    def mae(moving_volume: np.ndarray, reference_volume: np.ndarray):
        return np.mean(np.abs(moving_volume - reference_volume))
    
    import pickle

    # First 15 Minute Experiment
    # learning_rates = [1.0]
    # n_samples = [3]
    # relaxation_factors = [0.99]
    # smoothing_sigmas = [0]
    # iterations = [100]
    # window_min_max = [(0, 80), (20, 60), (-100, 300)]
    # interpolators = [sitk.sitkLinear, sitk.sitkBSpline]
    # ref_index = [0, 1, 4]
    # multi_res = [True]
    # mask = [True]
    
    # learning_rates = [1.0]
    # n_samples = [3]
    # relaxation_factors = [0.99]
    # smoothing_sigmas = [0]
    # iterations = [100]
    # window_min_max = [(0, 80)]
    # interpolators = [sitk.sitkLinear]
    # ref_index = [4]
    # multi_res = [True]
    # mask = [False, True]
    
    # learning_rates = [1.0]
    # n_samples = [3]
    # relaxation_factors = [0.99]
    # smoothing_sigmas = [0]
    # iterations = [100]
    # window_min_max = [(0, 80)]
    # interpolators = [sitk.sitkLinear]
    # ref_index = [4]
    # multi_res = [False, True]
    # mask = [True]
    # --> Multi res was better

    learning_rates = [0.5, 1.0]
    n_samples = [3]
    relaxation_factors = [0.99]
    smoothing_sigmas = [0]
    iterations = [150]
    window_min_max = [(0, 80)]
    interpolators = [sitk.sitkLinear]
    ref_index = [2] # [2, 4, 8] without temporal downsampling
    multi_res = [True]
    mask = [True]
    shrink_factors = [[2, 1], [2, 1]]
    smoothing_sigmas = [[2, 1], [2, 0]]
    # --> ref_index = 2 was best

    scan_ids = ["/MOL-" + s for s in ['063', '092']]
    # ---------------------------VOLUME LOADING--------------------------------
    import random
    all_paths = load_folder_paths(scan_size='small')
    # paths = [dataset_path + sid for sid in scan_ids] + random.sample(all_paths, 5)
    paths = random.sample(all_paths, 5)
    print("loading scans")
    # if os.path.exists("registration_scans.pkl"):
    #     with open("registration_scans.pkl", "rb") as f:
    #         scans = pickle.load(f)[:len(scan_ids)]
    # else:
    scans = [get_volume(path, 
            extract_brain=False, 
            filter=True, 
            window_params=None, 
            correct_motion=False, 
            standardize=False, 
            spatial_downsampling_factor=2, 
            temporal_downsampling_factor=2,
            verbose=False
            ) for path in paths]    

        # with open("registration_scans.pkl", "wb") as f:
        #     pickle.dump(scans, f)
    # ---------------------------PRINTS--------------------------------
    print(f"{len(scans)}, {len(scans[0])}, {len(learning_rates)}, {len(iterations)}, {len(n_samples)}, {len(relaxation_factors)}, {len(smoothing_sigmas)}, {len(interpolators)}, {len(mask)}, {len(window_min_max)}, {len(ref_index)}, {len(multi_res)}")
    total = 7 * len(learning_rates) * len(iterations) * len(n_samples) * len(relaxation_factors) * len(smoothing_sigmas) * \
        len(interpolators) * len(mask) * len(window_min_max) * len(ref_index) * len(multi_res) * len(scans) * len(scans[0])
    print(f"Total experiments: {total}")
    print(f"time eta: {total * 3 / 60} minutes")

    def write_results(results):
        with open("reg_exp_results.pkl", "wb") as f:
            pickle.dump(results, f)
    # ---------------------------RUNNING EXPERIMENTS--------------------------------
    print("running experiments")
    results = []
    from time import time
    start_time = time()
    # Experiment
    counter = 0
    for lr in learning_rates:
        # for i in iterations:
        for n in n_samples:
            for relax in relaxation_factors:
                for m_r in multi_res:
                    sigma = 0 if m_r else 1.0
                    # i = 150 if m_r else 300
                    # for sigma in smoothing_sigmas if not multi_res else [0]:
                    for interp in interpolators:
                        for (window_min, window_max) in window_min_max:
                            for m in mask:
                                for multi_res_index in range(len(shrink_factors)):
                                    i = 150
                                    for ref_i in ref_index:
                                        # Mean over all sequences
                                        total_mean_metric = 0
                                        # For more information about sequence-specific metrics
                                        total_metrics = {}
                                        for v_seq, seq_id in zip(scans, scan_ids):
                                            registered_seq = rigid_register_volume_sequence(v_seq, ref_i, multi_res=m_r, mask=m, shrink_factors=shrink_factors[multi_res_index], smoothing_sigmas=smoothing_sigmas[multi_res_index], verbose=False)
                                            # registered_seq = np.zeros_like(v_seq)
                                            # registered_seq[ref_i] = v_seq[ref_i]
                                            # # Metric for each volume in the sequence
                                            # sequence_metrics = []
                                            # # Mean over all volumes in the sequence
                                            # sequence_mean_metric = 0

                                            # v_ref = v_seq[ref_i]
                                            # for moving_idx in range(len(v_seq)):
                                            #     if moving_idx == ref_i:
                                            #         continue
                                            #     registered = experiment_reg(
                                            #                                 v_seq[moving_idx], 
                                            #                                 v_ref, 
                                            #                                 lr=lr, 
                                            #                                 n_iters=i,
                                            #                                 n_samples=n, 
                                            #                                 relaxation_factor=relax,
                                            #                                 multi_res=m_r,
                                            #                                 smoothing_sigma=sigma, 
                                            #                                 interpolator=interp,
                                            #                                 mask=m,
                                            #                                 reg_window_min=window_min,
                                            #                                 reg_window_max=window_max,
                                            #                                 shrink_factors=shrink_factors[multi_res_index],
                                            #                                 smoothing_sigmas=smoothing_sigmas[multi_res_index]
                                            #                                 )
                                            #     registered_seq[moving_idx] = registered
                                            #     metric = mse(np.clip(registered, -10, 60), np.clip(v_ref, -10, 60))
                                            #     sequence_metrics.append(metric)
                                            #     sequence_mean_metric += metric / (len(v_seq) - 1)

                                                # counter += 1
                                            save_volume(registered_seq, f"TestScans/{multi_res_index+1}/{seq_id}.npy")
                                            # total_metrics[seq_id] = (sequence_mean_metric, np.std(sequence_metrics))
                                            # total_mean_metric += sequence_mean_metric / len(scan_ids)
                                            total_metrics[seq_id] = mse(registered_seq, scans[0])
                                            total_mean_metric += mse(registered_seq, scans[0]) / len(scan_ids)

                                        result = {
                                                'total_mean_metric': total_mean_metric,
                                                'total_metrics': total_metrics,
                                                'reference_index': ref_i, 
                                                'learning_rate': lr, 
                                                'iterations': i,
                                                'n_samples': n, 
                                                'relaxation_factor': relax, 
                                                'multi_res': m_r, 
                                                'smoothing_sigma': sigma, 
                                                'interpolator': interp, 
                                                'mask': m, 
                                                'window_params': (window_min, window_max)
                                                }
                                        results.append(result)
                                        if counter < 20:
                                            print(result)
                                    # print(f'update: {counter / total * 100:.1f}% done')
                                    # print(f"time eta: {(time() - start_time) * (total / counter - 1) / 60:.4f} minutes")
                        # write_results(results)
    end_time = time()
    print(f"Time taken: {end_time - start_time} / 60 minutes")
    print([(r['total_mean_metric'], r['total_metrics'], ) for r in results])