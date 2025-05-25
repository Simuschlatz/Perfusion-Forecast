import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """
    DoubleConv3D
    -------------
    Applies two successive 3D convolutions with Batch Normalization and ReLU activation.

    Input shape: (B, in_channels, D, H, W)
    Output shape: (B, out_channels, D, H, W)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),  # -> (B, mid_channels, D, H, W)
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),  # -> (B, out_channels, D, H, W)
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
    
class AdvancedTemporalBlock3D(nn.Module):
    """
    AdvancedTemporalBlock3D
    -------------------------
    This block processes temporal relationships on volumetric features with enhanced sophistication.
    
    It applies a series of 1D convolutions along the temporal dimension with:
      - Increasing dilation rates (to widen the receptive field)
      - Residual connections (to ease optimization)
      - Dropout (for regularization)
      - Batch normalization and ReLU for improved training stability
      
    Input shape: (B, C, T, D, H, W)
    Output shape: (B, C, T, D, H, W)  (the temporal length T is preserved)
    """
    def __init__(self, channels, kernel_size, num_layers=2, dropout=0.2, dilation_base=2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for i in range(num_layers):
            dilation = dilation_base ** i   # Increase dilation at each layer
            # Set padding so that output temporal size equals input T.
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, x):
        # Input x shape: (B, C, T, D, H, W)
        B, C, T, D, H, W = x.shape
        
        # Reshape to merge the spatial dimensions: (B*D*H*W, C, T)
        x_reshaped = x.permute(0, 3, 4, 5, 1, 2).contiguous().view(-1, C, T)
        
        # Apply a series of temporal convolutions with residual connections.
        for layer in self.layers:
            out = layer(x_reshaped)  # -> (B*D*H*W, C, T)
            out = self.dropout(out)
            # Residual connection: the output of each layer is added to its input.
            x_reshaped = x_reshaped + out
        
        # Restore the original shape:
        # First reshape back to (B, D, H, W, C, T)
        x_reshaped = x_reshaped.view(B, D, H, W, C, T)
        # Permute back to (B, C, T, D, H, W)
        out = x_reshaped.permute(0, 4, 5, 1, 2, 3).contiguous()
        return out
    
    
class UNet3DPlusTemporal(nn.Module):
    """
    UNet3DPlusTemporal with Advanced Temporal Blocks
    --------------------------------------------------
    Extended UNet designed for volumetric (3D) data with temporal sequences.
    
    Expected input shape: (B, C, T, D, H, W)
      B = batch size
      C = number of input channels (e.g., 1 for grayscale)
      T = number of time frames (must equal input_frames)
      D, H, W = spatial dimensions
      
    Spatial processing is performed using 3D convolutions on each time step independently.
    Temporal processing is then applied on the stacked features using the advanced temporal blocks.
    """
    def __init__(self, input_frames=8, base_filters=32, in_channels=1):
        super().__init__()
        self.input_frames = input_frames
        
        # ---------------------
        # Encoder Path
        # ---------------------
        # Process each 3D volume (at a given time step) independently.
        self.enc1 = DoubleConv3D(in_channels, base_filters)  
        # After enc1: (B, base_filters, D, H, W)
        self.pool1 = nn.MaxPool3d(2)  # -> (B, base_filters, D/2, H/2, W/2)
        
        self.enc2 = DoubleConv3D(base_filters, base_filters * 2)
        # After enc2: (B, base_filters*2, D/2, H/2, W/2)
        self.pool2 = nn.MaxPool3d(2)  # -> (B, base_filters*2, D/4, H/4, W/4)
        
        self.enc3 = DoubleConv3D(base_filters * 2, base_filters * 4)
        # After enc3: (B, base_filters*4, D/4, H/4, W/4)
        # Apply an advanced temporal block to capture temporal dynamics before spatial pooling.
        self.temporal_enc = AdvancedTemporalBlock3D(channels=base_filters * 4, kernel_size=3, num_layers=2, dropout=0.2, dilation_base=2)
        self.pool3 = nn.MaxPool3d(2)  # -> (B, base_filters*4, D/8, H/8, W/8)
        
        # ---------------------
        # Bottleneck
        # ---------------------
        self.bottleneck_spatial = DoubleConv3D(base_filters * 4, base_filters * 8)
        # Bottleneck features: (B, base_filters*8, D/8, H/8, W/8)
        self.temporal_bottleneck = AdvancedTemporalBlock3D(channels=base_filters * 8, kernel_size=3, num_layers=2, dropout=0.2, dilation_base=2)
        
        # ---------------------
        # Decoder Path
        # ---------------------
        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_filters * 2, base_filters)
        
        self.final_conv = nn.Conv3d(base_filters, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass of UNet3DPlusTemporal.
        
        x: Input tensor of shape (B, T, C, D, H, W)
        """
        B, T, C, D, H, W = x.shape
        assert T == self.input_frames, f"Expected {self.input_frames} frames, got {T}"
        
        encoder_features = []  # To store skip connections from earlier encoder stages.
        enc3_features = []     # To store features from enc3 for temporal processing.
        
        # Process each time step independently through the encoder.
        for i in range(T):
            # Extract the 3D volume for time step i: (B, C, D, H, W)
            curr_vol = x[:, i]
            
            # Encoder Stage 1
            e1 = self.enc1(curr_vol)         # -> (B, base_filters, D, H, W)
            p1 = self.pool1(e1)              # -> (B, base_filters, D/2, H/2, W/2)
            
            # Encoder Stage 2
            e2 = self.enc2(p1)               # -> (B, base_filters*2, D/2, H/2, W/2)
            p2 = self.pool2(e2)              # -> (B, base_filters*2, D/4, H/4, W/4)
            
            encoder_features.append((e1, e2))  # Save skip connection features for the decoder.
            
            # Encoder Stage 3
            e3 = self.enc3(p2)               # -> (B, base_filters*4, D/4, H/4, W/4)
            enc3_features.append(e3)
        
        # Stack enc3 features along the temporal dimension.
        # New shape: (B, base_filters*4, T, D/4, H/4, W/4)
        enc3_features = torch.stack(enc3_features, dim=2)
        
        # Apply advanced temporal processing.
        enc3_processed = self.temporal_enc(enc3_features)  # -> (B, base_filters*4, T, D/4, H/4, W/4)
        
        # Spatial pooling across the processed features.
        B, C3, T, D_enc, H_enc, W_enc = enc3_processed.shape
        enc3_pooled = enc3_processed.view(B * T, C3, D_enc, H_enc, W_enc)  # (B*T, C3, D/4, H/4, W/4)
        enc3_pooled = self.pool3(enc3_pooled)   # -> (B*T, C3, D/8, H/8, W/8)
        _, _, D_pool, H_pool, W_pool = enc3_pooled.shape
        enc3_pooled = enc3_pooled.view(B, C3, T, D_pool, H_pool, W_pool)
        
        # Bottleneck processing: apply the spatial bottleneck then advanced temporal processing for each time step.
        bottle_features = []
        for i in range(T):
            feat = enc3_pooled[:, :, i]  # -> (B, C3, D/8, H/8, W/8)
            bottle_feat = self.bottleneck_spatial(feat)  # -> (B, base_filters*8, D/8, H/8, W/8)
            bottle_features.append(bottle_feat)
        # Stack to get shape: (B, base_filters*8, T, D/8, H/8, W/8)
        bottle_features = torch.stack(bottle_features, dim=2)
        bottle_processed = self.temporal_bottleneck(bottle_features)  # -> (B, base_filters*8, T, D/8, H/8, W/8)
        
        # For decoding, we select the final temporal state.
        #bottle_final = bottle_processed[:, :, -1]  # -> (B, base_filters*8, D/8, H/8, W/8)
        
        # ---------------------
        # Decoder Path
        # ---------------------
        # Retrieve skip connection features from the last time step.
        
        #e1_last, e2_last = encoder_features[-1]
        #e3_last = enc3_processed[:, :, -1]  # -> (B, base_filters*4, D/4, H/4, W/4)
        decoder_features = []
        for t in range(T):
            e1_last, e2_last = encoder_features[t]
            bottle_final = bottle_processed[:, :, t]
            e3_last = enc3_processed[:, :, t]
            
            d3 = self.upconv3(bottle_final)   # -> (B, base_filters*4, D/4, H/4, W/4)
            d3 = torch.cat([d3, e3_last], dim=1)  # Concatenated channels: (B, base_filters*8, D/4, H/4, W/4)
            d3 = self.dec3(d3)  # -> (B, base_filters*4, D/4, H/4, W/4)
            
            d2 = self.upconv2(d3)             # -> (B, base_filters*2, D/2, H/2, W/2)
            d2 = torch.cat([d2, e2_last], dim=1)   # -> (B, base_filters*4, D/2, H/2, W/2)
            d2 = self.dec2(d2)  # -> (B, base_filters*2, D/2, H/2, W/2)
            
            d1 = self.upconv1(d2)             # -> (B, base_filters, D, H, W)
            d1 = torch.cat([d1, e1_last], dim=1)   # -> (B, base_filters*2, D, H, W)
            d1 = self.dec1(d1)  # -> (B, base_filters, D, H, W)
            
            d1 = self.final_conv(d1)         # -> (B, 1, D, H, W)
            decoder_features.append(d1)
        decoder_features = torch.stack(decoder_features, dim=1)
        return decoder_features