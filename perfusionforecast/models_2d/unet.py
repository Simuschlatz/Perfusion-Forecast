import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """
    This block applies two consecutive 3D convolutions (with 3×3×3 kernels and
    padding=1 so that spatial dimensions are preserved) followed by ReLU activations.
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
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
        super(UNet3D, self).__init__()
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 32, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        
        self.enc2 = DoubleConv3D(64, 64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        
        self.enc3 = DoubleConv3D(128, 128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2))
        
        self.enc4 = DoubleConv3D(256, 256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=(1,2,2), stride=(1,2,2))
        # After upsampling: concatenation of upsampled features and corresponding encoder feature (256+256)
        self.dec3 = DoubleConv3D(256 + 256, 256, 256)
        
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = DoubleConv3D(128 + 128, 128, 128)
        
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = DoubleConv3D(64 + 64, 64, 64)
        
        self.out_conv = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x: tensor of shape (B, input_frames, C, H, W)
        
        Returns:
            Output tensor of shape (B, T, output_frames, H, W)
        """
        # Permute input from (B, T, C, H, W) to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
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
        # Permute output back to (B, T, n_classes, H, W) to match input convention
        out = out.permute(0, 2, 1, 3, 4)
        return out