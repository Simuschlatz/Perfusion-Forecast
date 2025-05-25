import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

class HuberSSIMLoss2D(nn.Module):
    def __init__(self, alpha=0.7, delta=0.05, window_size=5, temporal_weight=0.1):
        super().__init__()
        # ------------------------------------------------------------------------------
        # ALPHA (Controls balance between Huber Loss and SSIM)
        # More alpha  → Model focuses on voxel accuracy (Huber loss)
        # Less alpha  → Model prioritizes structural similarity (SSIM)
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - 0.6 - 0.9 → Noisy data (CT/MRI with artifacts) → More Huber
        # - 0.4 - 0.7 → Sharp structures (CT/MRI edges)  → Balance of both
        # - 0.3 - 0.5 → Blurry predictions → More SSIM for finer details
        # Default: alpha = 0.7 (Strong Huber, some SSIM)
        # ------------------------------------------------------------------------------
        self.alpha = alpha  # More alpha = More reliance on Huber
        # ------------------------------------------------------------------------------
        # DELTA (Threshold where Huber Loss switches from MSE-like to MAE-like behavior)
        # - Higher delta → More sensitive to small errors (acts like MSE)
        # - Lower delta  → More resistant to outliers (acts like MAE)
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - > 0.05  → High-noise data (CT/MRI with artifacts) → More robust to outliers
        # - < 0.05  → Low-noise data (Well-normalized, synthetic) → More sensitive to details
        # - 0.02 - 0.05  → If predictions are too blurry
        # Default: delta = 0.05 (or dynamically adjusted per epoch)
        # ------------------------------------------------------------------------------
        self.delta = delta  # This will be dynamically updated every epoch
        # ------------------------------------------------------------------------------
        # TEMPORAL WEIGHT (Penalizes abrupt voxel intensity changes between frames)
        # - Higher value → Forces smoother transitions
        # - Lower value  → Allows more flexibility in voxel changes
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - > 0.2  → Strong penalty for sudden intensity jumps (Flickering reduction)
        # - 0.05 - 0.1  → Best for smoothly changing sequences (MRI, Weather, Fluids)
        # - < 0.05  → Allows more dynamic changes (if model is too rigid)
        # Default: temporal_weight = 0.1 (balanced smoothness)
        # ------------------------------------------------------------------------------
        self.temporal_weight = temporal_weight  # More weight = Stronger smoothness enforcement

        #window_size means padding
        self.ssim_module = kornia.losses.SSIMLoss(window_size=window_size, reduction="mean")

    def temporal_smoothness_loss(self, y_pred):
        #Penalizes sudden changes over time by computing L1 loss between consecutive frames.
        #So the object should remain stationary over time
        return torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))  # Difference between t and t+1

    def forward(self, y_pred, y_true):
        B, T, C, H, W = y_pred.shape

        # Swap time and channel dimensions: [B, C*T, H, W]
        y_pred_restructured = y_pred.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        y_true_restructured = y_true.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)

        # Computes the SSIM treating time as another spatial dimension
        ssim_loss = self.ssim_module(y_pred_restructured, y_true_restructured)

        # Computes the Huber loss with the adjusted delta
        huber_loss = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="mean")

        # Compute the Temporal Smoothness Loss
        temporal_loss = self.temporal_smoothness_loss(y_pred)

        # Weighted combination
        total_loss = self.alpha * huber_loss + (1 - self.alpha) * ssim_loss + self.temporal_weight * temporal_loss
        return total_loss, huber_loss, ssim_loss, temporal_loss

class HuberSSIMLoss3D(nn.Module):
    def __init__(self, alpha=0.7, delta=0.05, window_size=5, temporal_weight=0.1):
        super().__init__()
        # ------------------------------------------------------------------------------
        # ALPHA (Controls balance between Huber Loss and SSIM)
        # More alpha  → Model focuses on voxel accuracy (Huber loss)
        # Less alpha  → Model prioritizes structural similarity (SSIM)
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - 0.6 - 0.9 → Noisy data (CT/MRI with artifacts) → More Huber
        # - 0.4 - 0.7 → Sharp structures (CT/MRI edges)  → Balance of both
        # - 0.3 - 0.5 → Blurry predictions → More SSIM for finer details
        # Default: alpha = 0.7 (Strong Huber, some SSIM)
        # ------------------------------------------------------------------------------
        self.alpha = alpha  # More alpha = More reliance on Huber
        # ------------------------------------------------------------------------------
        # DELTA (Threshold where Huber Loss switches from MSE-like to MAE-like behavior)
        # - Higher delta → More sensitive to small errors (acts like MSE)
        # - Lower delta  → More resistant to outliers (acts like MAE)
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - > 0.05  → High-noise data (CT/MRI with artifacts) → More robust to outliers
        # - < 0.05  → Low-noise data (Well-normalized, synthetic) → More sensitive to details
        # - 0.02 - 0.05  → If predictions are too blurry
        # Default: delta = 0.05 (or dynamically adjusted per epoch)
        # ------------------------------------------------------------------------------
        self.delta = delta  # This will be dynamically updated every epoch
        # ------------------------------------------------------------------------------
        # TEMPORAL WEIGHT (Penalizes abrupt voxel intensity changes between frames)
        # - Higher value → Forces smoother transitions
        # - Lower value  → Allows more flexibility in voxel changes
        # ------------------------------------------------------------------------------
        # Recommended tuning:
        # - > 0.2  → Strong penalty for sudden intensity jumps (Flickering reduction)
        # - 0.05 - 0.1  → Best for smoothly changing sequences (MRI, Weather, Fluids)
        # - < 0.05  → Allows more dynamic changes (if model is too rigid)
        # Default: temporal_weight = 0.1 (balanced smoothness)
        # ------------------------------------------------------------------------------
        self.temporal_weight = temporal_weight  # More weight = Stronger smoothness enforcement

        #window_size means padding
        self.ssim_module = kornia.losses.SSIM3DLoss(window_size=window_size, reduction="mean")

    def temporal_smoothness_loss(self, y_pred):
        #Penalizes sudden changes over time by computing L1 loss between consecutive frames.
        #So the object should remain stationary over time
        return torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))  # Difference between t and t+1

    def forward(self, y_pred, y_true):
        B, T, C, D, H, W = y_pred.shape

        # Swap time and channel dimensions: [B, C*T, D, H, W]
        y_pred_restructured = y_pred.permute(0, 2, 1, 3, 4, 5).reshape(B, C * T, D, H, W)
        y_true_restructured = y_true.permute(0, 2, 1, 3, 4, 5).reshape(B, C * T, D, H, W)

        # Computes the SSIM treating time as another spatial dimension
        ssim_loss = self.ssim_module(y_pred_restructured, y_true_restructured)

        # Computes the Huber loss with the adjusted delta
        huber_loss = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="mean")

        # Compute the Temporal Smoothness Loss
        temporal_loss = self.temporal_smoothness_loss(y_pred)

        # Weighted combination
        total_loss = self.alpha * huber_loss + (1 - self.alpha) * ssim_loss + self.temporal_weight * temporal_loss
        return total_loss, huber_loss, ssim_loss, temporal_loss

@torch.no_grad()
def overall_loss_2d(model, loader, device):
    mse_loss = 0.0
    huber_loss = 0.0
    rmse_loss = 0.0
    #ssim_loss = 0.0
    psnr_loss = 0.0
    total_loss = 0.0
    model = model.to(device)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # iterate over depth
        outputs = []
        for d in range(inputs.shape[3]):
            inputs_ = inputs[:, :, :, d, :, :]
            #iterate over time
            outputs_d = []
            for t in range(0, model.config["pred_frames"], model.config["pred_n_frames_per_step"]):
                if model.config["pred_frames"]-t<model.config["pred_n_frames_per_step"]:
                    frames_this_step = model.config["pred_frames"]-t
                else:
                    frames_this_step = model.config["pred_n_frames_per_step"]
                
                outputs_t = model.forward(inputs_)
        
                #get only the first predicted frame
                outputs_t = outputs_t[:, :frames_this_step, :, :]

                #add to depth lst
                outputs_d.append(outputs_t)
        
                inputs_ = torch.cat([inputs_[:, model.config["pred_n_frames_per_step"]:, :, :], outputs_t], dim=1)
            #concat time and add to overall lst
            outputs_d = torch.concat(outputs_d, dim=1)
            outputs.append(outputs_d)
    
        #stack over depth
        outputs = torch.stack(outputs, dim=3)
        #print(outputs.shape)
        
        #calculate losses
        mse_loss += model.mse_criterion(outputs, targets).item()
        huber_loss += model.huber_criterion(outputs, targets).item()
        rmse_loss += torch.sqrt(model.mse_criterion(outputs, targets)).item()
        #ssim_loss = model.ssim_criterion(outputs, targets).item()
        psnr_loss += model.psnr_criterion(outputs, targets).item()
        total_loss += huber_loss

    mse_loss = mse_loss / len(loader)
    huber_loss = huber_loss / len(loader)
    rmse_loss = rmse_loss / len(loader)
    #ssim_loss = ssim_loss / len(loader)
    psnr_loss = psnr_loss / len(loader)
    total_loss = total_loss / len(loader)

    return outputs, mse_loss, huber_loss, rmse_loss, psnr_loss, total_loss

@torch.no_grad()
def overall_loss_3d(model, loader, device):
    mse_loss = 0.0
    huber_loss = 0.0
    rmse_loss = 0.0
    #ssim_loss = 0.0
    psnr_loss = 0.0
    total_loss = 0.0
    model = model.to(device)
    for inputs, targets in loader:
        print(targets.shape)
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = []
        for t in range(0, model.config["pred_frames"], model.config["pred_n_frames_per_step"]):
            if model.config["pred_frames"]-t<model.config["pred_n_frames_per_step"]:
                frames_this_step = model.config["pred_frames"]-t
            else:
                frames_this_step = model.config["pred_n_frames_per_step"]
            outputs_t = model.forward(inputs)
            #print(f"{t}:{t+frames_this_step}")
            #get only the first predicted frame
            outputs_t = outputs_t[:, :frames_this_step, :, :, :, :]
            
            outputs.append(outputs_t)

            inputs = torch.cat([inputs[:, model.config["pred_n_frames_per_step"]:, :, :, :, :], outputs_t], dim=1)
            
            #concat time and add to overall lst
        outputs = torch.concat(outputs, dim=1)
        
        #calculate losses
        mse_loss += model.mse_criterion(outputs, targets).item()
        huber_loss += model.huber_criterion(outputs, targets).item()
        rmse_loss += torch.sqrt(model.mse_criterion(outputs, targets)).item()
        #ssim_loss = model.ssim_criterion(outputs, targets).item()
        psnr_loss += model.psnr_criterion(outputs, targets).item()
        total_loss += mse_loss + 0.5 * huber_loss

    mse_loss = mse_loss / len(loader)
    huber_loss = huber_loss / len(loader)
    rmse_loss = rmse_loss / len(loader)
    #ssim_loss = ssim_loss / len(loader)
    psnr_loss = psnr_loss / len(loader)
    total_loss = total_loss / len(loader)

    return outputs, mse_loss, huber_loss, rmse_loss, psnr_loss, total_loss