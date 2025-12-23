import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==============================================================================
# 1. Utility Function (from your notebook)
# ==============================================================================

def compute_fft_batch(images):
    """Computes the FFT (frequency) representation of a batch of images."""
    # Convert RGB to grayscale: (B, C, H, W) -> (B, H, W)
    gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    gray = gray.unsqueeze(1)  # Shape: (B, 1, H, W)

    # Compute FFT
    f = torch.fft.fft2(gray)
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.abs(f).clamp_min(1e-8)
    log_mag = torch.log(mag)

    # Normalize
    b, c, h, w = log_mag.shape
    log_mag_flat = log_mag.view(b, c, -1)
    mn = log_mag_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
    mx = log_mag_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
    freq = (log_mag - mn) / (mx - mn + 1e-8)

    return freq  # Shape: (B, 1, H, W)

# ==============================================================================
# 2. Frame-Level Model Definitions (from your notebook)
# ==============================================================================

class MultiScaleFrequencyBranch(nn.Module):
    """Multi-scale frequency analysis - detects artifacts at different frequencies"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.low_freq = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.high_freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=2, padding=1),  # 64*3=192
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        low = self.low_freq(x)
        mid = self.mid_freq(x)
        high = self.high_freq(x)

        h, w = mid.shape[2], mid.shape[3]
        low = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        high = F.interpolate(high, size=(h, w), mode='bilinear', align_corners=False)

        multi_scale = torch.cat([low, mid, high], dim=1)
        fused = self.fusion(multi_scale).flatten(1)
        return self.fc(fused)

class SpatialAttention(nn.Module):
    """Attention mechanism to focus on important facial regions"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class EnhancedDeepFakeDetector(nn.Module):
    """The Dual-Branch Frame-Level Model"""
    def __init__(self, freeze_backbone=True, spatial_dim=512, freq_dim=256):
        super().__init__()
        # Note: In the app, weights will be loaded from the .pt file,
        # so 'weights=...' is only for architecture definition.
        efficientnet = models.efficientnet_b0(weights=None) 

        if freeze_backbone:
            for param in list(efficientnet.features.parameters())[:-20]:
                param.requires_grad = False

        self.spatial = efficientnet.features
        self.spatial_attention = SpatialAttention(1280)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_fc = nn.Sequential(
            nn.Linear(1280, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(spatial_dim, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.freq_branch = MultiScaleFrequencyBranch(freq_dim)
        self.cross_attention = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spatial_dim + freq_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, spatial_input, freq_input):
        # Spatial pathway
        s = self.spatial(spatial_input)
        s = self.spatial_attention(s)
        s = self.spatial_pool(s).flatten(1)
        s = self.spatial_fc(s)

        # Frequency pathway
        f = self.freq_branch(freq_input)

        # Concatenate features
        combined = torch.cat([s, f], dim=1)

        # Cross-attention
        attention_weights = self.cross_attention(combined)
        combined = combined * attention_weights

        # Final classification
        # Note: In the Full model, this 'fusion' layer is replaced
        # and this output is the 768-dim feature vector
        output = self.fusion(combined)
        return output

# ==============================================================================
# 3. Video-Level Model Definitions (from your notebook)
# ==============================================================================

class TemporalBiLSTM(nn.Module):
    """The LSTM head that reads the sequence of frame features."""
    def __init__(self, feature_dim=768, hidden_dim=512, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape is (B, T, feature_dim)
        out, _ = self.lstm(x)
        # Get the mean of all frame outputs
        pooled = torch.mean(out, dim=1) 
        return self.fc(pooled)

class FullDeepFakeDetector(nn.Module):
    """
    The FINAL Wrapper Model.
    Combines the frame-level model and the temporal model.
    """
    def __init__(self, base_weights_path=None, device='cpu'):
        super().__init__()
        
        # 1. Load the Frame-Level Model
        self.frame_model = EnhancedDeepFakeDetector(freeze_backbone=True).to(device)
        
        # When loading the final MarkV.pt, this init logic is skipped.
        # But for loading the *base* model, it's needed.
        if base_weights_path:
            state_dict = torch.load(base_weights_path, map_location=device)
            # Load weights, ignoring the final 'fusion' layer which we are replacing
            self.frame_model.load_state_dict(state_dict, strict=False)
        
        self.frame_model.eval()
        
        # 2. Replace the frame model's final classifier with an Identity
        # so it just outputs the 768-dim feature vector
        self.frame_model.fusion = nn.Identity()
        
        # 3. Create the Temporal (LSTM) head
        self.temporal = TemporalBiLSTM(feature_dim=768, hidden_dim=512)

    def forward(self, frames_tensor):
        """
        NEW, FASTER, VECTORIZED FORWARD PASS
        This is much faster than the original Python loop and
        enables Grad-CAM for explainability.
        
        Input shape: (B, T, C, H, W) 
                       e.g., (1, 10, 3, 224, 224)
        """
        b, t, c, h, w = frames_tensor.shape
        
        # 1. Reshape to process all frames as one big batch
        # (B, T, C, H, W) -> (B*T, C, H, W)
        frames_flat = frames_tensor.view(b * t, c, h, w)
        
        # 2. Get frequency features for the whole batch
        # (B*T, C, H, W) -> (B*T, 1, H, W)
        freq_in = compute_fft_batch(frames_flat)
        
        # 3. Get 768-dim features from the frozen frame model
        # This single call processes all frames from all videos in the batch
        # Output shape: (B*T, 768)
        with torch.no_grad(): # Ensure it's frozen
            features_flat = self.frame_model(frames_flat, freq_in)
        
        # 4. Reshape features back into a sequence
        # (B*T, 768) -> (B, T, 768)
        features_seq = features_flat.view(b, t, -1)
        
        # 5. Pass the sequence to the LSTM to get the final video-level prediction
        # Output shape: (B, 1)
        output = self.temporal(features_seq)
        
        return output