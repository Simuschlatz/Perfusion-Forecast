import torch
import torch.nn as nn
import torch.nn.functional as F

from .convNd import convNd


class MaxPool4d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride if stride is not None else kernel_size)

    def _to_tuple(self, x):
        if isinstance(x, int):
            return (x, x, x, x)
        elif len(x) == 4:
            return tuple(x)
        else:
            raise ValueError("kernel_size and stride must be int or 4-tuple")

    def forward(self, x):
        # x: (N, C, D1, D2, D3, D4)
        N, C, D1, D2, D3, D4 = x.shape
        k1, k2, k3, k4 = self.kernel_size
        s1, s2, s3, s4 = self.stride

        # Calculate output sizes
        out_D1 = (D1 - k1) // s1 + 1
        out_D2 = (D2 - k2) // s2 + 1
        out_D3 = (D3 - k3) // s3 + 1
        out_D4 = (D4 - k4) // s4 + 1

        # Use unfold to extract sliding windows in all 4 spatial dims
        x = x.unfold(2, k1, s1).unfold(3, k2, s2).unfold(4, k3, s3).unfold(5, k4, s4)
        # shape: (N, C, out_D1, out_D2, out_D3, out_D4, k1, k2, k3, k4)

        x = x.contiguous().view(N, C, out_D1, out_D2, out_D3, out_D4, -1)
        x, _ = x.max(dim=-1)  # max over the kernel volume

        return x


class DoubleConv4D(nn.Module):
    """
    This block applies two consecutive 3D convolutions (with 3×3×3 kernels and
    padding=1 so that spatial dimensions are preserved) followed by ReLU activations.
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv4D, self).__init__()
        self.double_conv = nn.Sequential(
            convNd(in_channels, mid_channels, num_dims=4, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            convNd(mid_channels, out_channels, num_dims=4, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet4D(nn.Module):
    """
    3D U-Net model with a four-layer encoder–decoder architecture.
    
    Adapted from Ichikawa, Shota, Makoto Ozaki, Hideki Itadani, Hiroyuki Sugimori, and Yohan Kondo, 
    ‘Deep Learning-Based Correction for Time Truncation in Cerebral Computed Tomography Perfusion’, 
    Radiological Physics and Technology, 17.3 (2024), pp. 666–78, doi:10.1007/s12194-024-00818-6

    Changes: omitted sigmoid activation function after final conv since data is standardized, not normalized
    (likely a mistake in the original paper)

    Expected input shape: (B, T, C, H, W) where C=1.

    Internally, we permute the input to (B, C, T, H, W) so that 3D operations work as intended.
    The encoder path consists of four blocks:
      - Block 1: Two convs from (in_channels → 32 → 64) followed by a max pool (1,2,2).
      - Block 2: Two convs (64 → 64 → 128) with max pool.
      - Block 3: Two convs (128 → 128 → 256) with max pool.
      - Block 4: Two convs (256 → 256 → 512) at the bottleneck.

    The decoder upsamples features using transposed convolutions (with kernel (1,2,2)),
    concatenates skip connections, and applies additional convs. Finally a 1×1×1 convolution
    maps to the desired number of output channels.

    The final output is permuted back to keep the temporal dimension as the second dimension.
    """
    def __init__(self, in_channels=1, n_classes=1):
        super(UNet4D, self).__init__()
        # Encoder
        self.enc1 = DoubleConv4D(in_channels, 16, 32)
        self.pool1 = MaxPool4d(kernel_size=(1,2,2,2))
        
        self.enc2 = DoubleConv4D(32, 32, 64)
        self.pool2 = MaxPool4d(kernel_size=(1,2,2,2))
        
        self.enc3 = DoubleConv4D(64, 64, 128)
        self.pool3 = MaxPool4d(kernel_size=(1,2,2,2))
        
        self.enc4 = DoubleConv4D(128, 128, 256)
        
        # Decoder
        self.up3 = convNd(256, 128, num_dims=4, kernel_size=(1,2,2,2), stride=(1,2,2,2), padding=0, is_transposed=True)
        # After upsampling: concatenation of upsampled features and corresponding encoder feature (256+256)
        self.dec3 = DoubleConv4D(128 + 128, 128, 128)
        
        self.up2 = convNd(128, 64, num_dims=4, kernel_size=(1,2,2,2), stride=(1,2,2,2), padding=0, is_transposed=True)
        self.dec2 = DoubleConv4D(64 + 64, 64, 64)
        
        self.up1 = convNd(64, 32, num_dims=4, kernel_size=(1,2,2,2), stride=(1,2,2,2), padding=0, is_transposed=True)
        self.dec1 = DoubleConv4D(32 + 32, 32, 32)
        
        self.out_conv = convNd(32, n_classes, num_dims=4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x: tensor of shape (B, input_frames, C, H, W)
        
        Returns:
            Output tensor of shape (B, T, output_frames, H, W)
        """
        # Permute input from (B, T, C, D, H, W) to (B, C, T, D, H, W) for Conv4D
        x = x.permute(0, 2, 1, 3, 4, 5)
        
        # Encoder path
        enc1 = self.enc1(x)        # => (B, 64, T, H, W)
        x = self.pool1(enc1)       # => (B, 64, T, H/2, W/2)
        
        enc2 = self.enc2(x)        # => (B, 128, T, H/2, W/2)
        x = self.pool2(enc2)       # => (B, 128, T, H/4, W/4)
        
        enc3 = self.enc3(x)        # => (B, 256, T, H/4, W/4)
        x = self.pool3(enc3)       # => (B, 256, T, H/8, W/8)
        
        x = self.enc4(x)           # => (B, 512, T, H/8, W/8)
        
        # Decoder path
        x = self.up3(x)            # => (B, 256, T, H/4, W/4)
        x = torch.cat([x, enc3], dim=1)  # Concatenate along channels: (B, 512, T, H/4, W/4)
        x = self.dec3(x)           # => (B, 256, T, H/4, W/4)
        
        x = self.up2(x)            # => (B, 128, T, H/2, W/2)
        x = torch.cat([x, enc2], dim=1)  # => (B, 256, T, H/2, W/2)
        x = self.dec2(x)           # => (B, 128, T, H/2, W/2)
        
        x = self.up1(x)            # => (B, 64, T, H, W)
        x = torch.cat([x, enc1], dim=1)  # => (B, 128, T, H, W)
        x = self.dec1(x)           # => (B, 64, T, H, W)
        
        out = self.out_conv(x)     # => (B, n_classes, T, H, W)
        # Permute output back to (B, T, n_classes, D, H, W) to match input convention
        out = out.permute(0, 2, 1, 3, 4, 5)
        
        return out
