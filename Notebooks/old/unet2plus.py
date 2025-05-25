import torch
import torch.nn as nn

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(
                in_channels, 
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False  # No bias when using batch norm
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False  # No bias when using batch norm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional: Add residual connection if input and output channels match
        self.use_residual = in_channels == out_channels
        
    def forward(self, x):
        if self.use_residual:
            return self.double_conv(x) + x
        return self.double_conv(x)
    
class TemporalBlock(nn.Module):
    def __init__(self, channels, temporal_kernel_size=3):
        super().__init__()
        
        padding = temporal_kernel_size // 2
        
        self.temporal_conv = nn.Sequential(
            # Depthwise temporal conv
            nn.Conv1d(channels, channels,
                     kernel_size=temporal_kernel_size,
                     padding=padding,
                     groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            
            # Point-wise conv
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        
        # Reshape for temporal convolution
        x_temp = x.contiguous()  # Make memory contiguous
        x_temp = x_temp.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
        x_temp = x_temp.reshape(b*h*w, c, t)
        
        # Apply temporal convolution
        x_temp = self.temporal_conv(x_temp)
        
        # Reshape back
        x_temp = x_temp.reshape(b, h, w, c, t)
        x_temp = x_temp.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
        
        return x_temp


class UNet2DPlusTemporal(nn.Module):
    def __init__(self, input_frames=None, base_filters=32):
        super().__init__()
        
        self.input_frames = input_frames
        
        # Encoder Path
        self.enc1 = DoubleConv2D(1, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv2D(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Last encoder with temporal processing
        self.enc3 = DoubleConv2D(base_filters*2, base_filters*4)
        self.temporal_enc = TemporalBlock(
            channels=base_filters*4,
            temporal_kernel_size=3
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck with temporal processing
        self.bottleneck_spatial = DoubleConv2D(base_filters*4, base_filters*8)
        self.temporal_bottleneck = TemporalBlock(
            channels=base_filters*8,
            temporal_kernel_size=3
        )
        
        # Decoder Path
        self.upconv3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 
                                         kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(base_filters*8, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 
                                         kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, 
                                         kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(base_filters*2, base_filters)
        
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch, time, height, width]
        b, t, h, w = x.shape
        assert t == self.input_frames, f"Expected {self.input_frames} frames, got {t}"
        
        # Process each time step through initial spatial encoders
        encoder_features = []
        enc3_features = []
        
        for i in range(t):
            curr_frame = x[:, i].unsqueeze(1)  # [B, 1, H, W]
            
            # Initial encoder path
            e1 = self.enc1(curr_frame)
            p1 = self.pool1(e1)
            
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)
            
            # Store for skip connections
            encoder_features.append((e1, e2))
            
            # Last encoder
            e3 = self.enc3(p2)
            enc3_features.append(e3)
        
        # Process enc3 features temporally
        enc3_features = torch.stack(enc3_features, dim=2)  # [B, C, T, H, W]
        enc3_processed = self.temporal_enc(enc3_features)
        
        # Pool spatially after temporal processing
        b, c, t, h, w = enc3_processed.shape
        enc3_pooled = enc3_processed.contiguous()
        enc3_pooled = enc3_pooled.view(b*t, c, h, w)
        enc3_pooled = self.pool3(enc3_pooled)
        _, _, h_pooled, w_pooled = enc3_pooled.shape
        enc3_pooled = enc3_pooled.view(b, c, t, h_pooled, w_pooled)
        
        # Bottleneck processing
        bottle_features = []
        for i in range(t):
            curr_feat = enc3_pooled[:, :, i]  # [B, C, H, W]
            bottle_feat = self.bottleneck_spatial(curr_feat)
            bottle_features.append(bottle_feat)
        
        # Stack and apply temporal processing in bottleneck
        bottle_features = torch.stack(bottle_features, dim=2)  # [B, C, T, H, W]
        bottle_processed = self.temporal_bottleneck(bottle_features)
        
        # Take last temporal state for decoder
        bottle_final = bottle_processed[:, :, -1]  # [B, C, H, W]
        
        # Decoder path (using last frame's encoder features)
        e1, e2 = encoder_features[-1]
        e3 = enc3_processed[:, :, -1]
        
        d3 = self.upconv3(bottle_final)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)


def test_model():
    # Test with different temporal dimensions
    temporal_sizes = [8, 16, 32]
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for T in temporal_sizes:
        print(f"\nTesting with T={T}")
        model = UNet2DPlusTemporal(input_frames=T).to(device)
        x = torch.randn(2, T, 256, 256).to(device)
        
        try:
            out = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print("Test passed!")
        except Exception as e:
            print(f"Error with T={T}: {str(e)}")
            raise e  # This will show the full error traceback

if __name__ == "__main__":
    test_model()