"""
Multi-Model Training Script with Configurable Backbones
Tests multiple architectures: ResNet50, AlexNet, MobileNetV2, and Hybrid models with different backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
import random
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import gc
from torch import amp

# Try to import pywt for MSAN model
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets (pywt) not installed. MSAN models will be skipped.")
    print("Install with: pip install PyWavelets")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
SCALER = amp.GradScaler('cuda', enabled=device.type == 'cuda')
SEED = 42

# Reproducibility helpers
def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # Ensures each worker has a different, deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 16
NUM_EPOCHS = 1
PATIENCE = 7
LEARNING_RATE = 0.0001
IMG_SIZE = 224
TRAIN_TEST_SPLIT = 0.2  # 20% test
VAL_SPLIT = 0.1         # 10% of train for validation (overall 72/8/20)

# ==================== FGCR MODULE ====================
class FrequencyGatedChannelRecalibration(nn.Module):
    """Frequency-Gated Channel Recalibration"""
    def __init__(self, channels, reduction=8, cutoff_ratio=0.25):
        super(FrequencyGatedChannelRecalibration, self).__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(2 * channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        magnitude = torch.abs(fft)

        mask_low, mask_high = self._frequency_masks(H, W, x.device, x.dtype)
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)

        energy_low = (magnitude * mask_low).mean(dim=(-2, -1))
        energy_high = (magnitude * mask_high).mean(dim=(-2, -1))

        stats = torch.cat([energy_low, energy_high], dim=-1)
        weights = self.fc2(self.act(self.fc1(stats)))
        weights = self.sigmoid(weights).view(B, C, 1, 1)

        recalibrated = x * weights
        spectral_token = torch.stack(
            [energy_low.mean(dim=1), energy_high.mean(dim=1)],
            dim=1
        )

        return recalibrated, spectral_token

    def _frequency_masks(self, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radial = torch.sqrt(xx ** 2 + yy ** 2)
        mask_low = (radial <= self.cutoff_ratio).float()
        mask_high = torch.ones_like(mask_low) - mask_low
        return mask_low, mask_high


# ==================== CSBA MODULE ====================
class CrossScaleBiAttention(nn.Module):
    """Cross-Scale Bi-Attention"""
    def __init__(self, c_low, c_high, attn_dim=256, num_heads=8, dropout=0.1):
        super(CrossScaleBiAttention, self).__init__()
        self.low_embed = nn.Linear(c_low, attn_dim)
        self.high_embed = nn.Linear(c_high, attn_dim)
        self.low_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.high_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.low_proj = nn.Linear(attn_dim, c_low)
        self.high_proj = nn.Linear(attn_dim, c_high)
        self.dropout = nn.Dropout(dropout)
        self.norm_low = nn.LayerNorm(c_low)
        self.norm_high = nn.LayerNorm(c_high)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(2 * attn_dim),
            nn.Linear(2 * attn_dim, attn_dim),
            nn.GELU()
        )
        self.token_dim = attn_dim

    def forward(self, low_feat, high_feat):
        B, C_low, H_low, W_low = low_feat.shape
        B, C_high, H_high, W_high = high_feat.shape

        low_tokens = low_feat.flatten(2).transpose(1, 2)
        high_tokens = high_feat.flatten(2).transpose(1, 2)

        low_emb = self.low_embed(low_tokens)
        high_emb = self.high_embed(high_tokens)

        high_ctx, _ = self.high_attn(high_emb, low_emb, low_emb)
        low_ctx, _ = self.low_attn(low_emb, high_emb, high_emb)

        high_tokens = self.norm_high(
            high_tokens + self.dropout(self.high_proj(high_ctx))
        )
        low_tokens = self.norm_low(
            low_tokens + self.dropout(self.low_proj(low_ctx))
        )

        high_out = high_tokens.transpose(1, 2).reshape(B, C_high, H_high, W_high)

        low_summary = low_ctx.mean(dim=1)
        high_summary = high_ctx.mean(dim=1)
        bridge_token = self.token_proj(torch.cat([low_summary, high_summary], dim=-1))

        return high_out, bridge_token


# ==================== BASELINE MODELS ====================
class BaselineBinary(nn.Module):
    """Baseline model without hybrid attention"""
    def __init__(self, backbone='resnet50', pretrained=True):
        super(BaselineBinary, self).__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        elif backbone == 'alexnet':
            self.model = models.alexnet(pretrained=pretrained)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, 1)
        elif backbone == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.model(x).squeeze(1)


# ==================== HYBRID FGCR+CSBA MODEL WITH CONFIGURABLE BACKBONE ====================
class HybridFGCR_CSBA(nn.Module):
    """
    Hybrid model with FGCR and CSBA, configurable backbone
    Supports: ResNet50, AlexNet, MobileNetV2
    """
    def __init__(self, backbone='resnet50', pretrained=True, num_heads=4, attn_dim=128, 
                 attn_dropout=0.2, cls_dropout=0.5):
        super(HybridFGCR_CSBA, self).__init__()
        self.backbone_name = backbone
        
        # Initialize backbone and get layer information
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3  # 1024 channels
            self.layer4 = resnet.layer4  # 2048 channels
            
            low_channels = 1024
            high_channels = 2048
            
        elif backbone == 'alexnet':
            alexnet = models.alexnet(pretrained=pretrained)
            # AlexNet doesn't have residual layers, we'll adapt the features
            self.features = alexnet.features
            # Add conv layers to match expected dimensions
            self.layer3_adapter = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.layer4_adapter = nn.Sequential(
                nn.Conv2d(256, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            low_channels = 1024
            high_channels = 2048
            
        elif backbone == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.features = mobilenet.features
            # MobileNetV2: features output is 1280 channels
            # We'll extract intermediate features
            self.layer3_adapter = nn.Sequential(
                nn.Conv2d(96, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU6(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU6(inplace=True)
            )
            self.layer4_adapter = nn.Sequential(
                nn.Conv2d(1280, 2048, kernel_size=3, padding=1),
                nn.BatchNorm2d(2048),
                nn.ReLU6(inplace=True)
            )
            
            low_channels = 1024
            high_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # FGCR modules
        self.fgcr3 = FrequencyGatedChannelRecalibration(low_channels)
        self.fgcr4 = FrequencyGatedChannelRecalibration(high_channels)
        
        # CSBA module
        self.cross_scale = CrossScaleBiAttention(
            c_low=low_channels,
            c_high=high_channels,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        classifier_in = high_channels + self.cross_scale.token_dim + 2
        self.classifier = nn.Sequential(
            nn.Dropout(cls_dropout),
            nn.Linear(classifier_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(512, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        modules = [self.fgcr3, self.fgcr4, self.cross_scale, self.classifier]
        if hasattr(self, 'layer3_adapter'):
            modules.append(self.layer3_adapter)
        if hasattr(self, 'layer4_adapter'):
            modules.append(self.layer4_adapter)
            
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if self.backbone_name == 'resnet50':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            
            x = self.layer3(x)
            x, _ = self.fgcr3(x)
            low_scale = x
            
            x = self.layer4(x)
            x, spectral_token = self.fgcr4(x)
            
        elif self.backbone_name == 'alexnet':
            x = self.features(x)  # [B, 256, H, W]
            
            low_scale = self.layer3_adapter(x)  # [B, 1024, H, W]
            low_scale, _ = self.fgcr3(low_scale)
            
            x = self.layer4_adapter(x)  # [B, 2048, H, W]
            x, spectral_token = self.fgcr4(x)
            
        elif self.backbone_name == 'mobilenet_v2':
            # Extract features at different stages
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 13:  # After block 6, 96 channels
                    low_scale = self.layer3_adapter(x)
                    low_scale, _ = self.fgcr3(low_scale)
            
            # Final features: 1280 channels
            x = self.layer4_adapter(x)
            x, spectral_token = self.fgcr4(x)
        
        # Cross-scale attention
        x, bridge_token = self.cross_scale(low_scale, x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Concatenate features
        stats = torch.cat([spectral_token, bridge_token], dim=1)
        logits = self.classifier(torch.cat([x, stats], dim=1))
        
        return logits.squeeze(1)


# ==================== MSAN MODEL COMPONENTS ====================
# Only include if pywt is available

if PYWT_AVAILABLE:
    # Spatial Transformer Network for MSAN
    class SpatialTransformerNetwork(nn.Module):
        """Learns to focus on specific anatomical regions"""
        def __init__(self, in_channels):
            super(SpatialTransformerNetwork, self).__init__()
            self.localization = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            
            self.fc_loc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, 6)
            )
            
            self.fc_loc[4].weight.data.zero_()
            self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        def forward(self, x):
            xs = self.localization(x)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
            
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            
            return x, theta


    class AnatomicalRegionProposal(nn.Module):
        """Proposes attention masks for anatomical regions"""
        def __init__(self, in_channels, num_regions=5):
            super(AnatomicalRegionProposal, self).__init__()
            self.num_regions = num_regions
            
            self.region_transformers = nn.ModuleList([
                SpatialTransformerNetwork(in_channels) 
                for _ in range(num_regions - 1)
            ])
            
            self.region_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True)
                ) for _ in range(num_regions)
            ])
            
            self.mask_generators = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 1, kernel_size=1),
                    nn.Sigmoid()
                ) for _ in range(num_regions)
            ])
        
        def forward(self, x):
            B, C, H, W = x.shape
            region_features = []
            attention_masks = []
            
            for i in range(self.num_regions):
                if i < self.num_regions - 1:
                    transformed_x, theta = self.region_transformers[i](x)
                    region_feat = self.region_extractors[i](transformed_x)
                else:
                    region_feat = self.region_extractors[i](x)
                
                mask = self.mask_generators[i](region_feat)
                masked_feat = region_feat * mask
                
                region_features.append(masked_feat)
                attention_masks.append(mask)
            
            region_features = torch.stack(region_features, dim=1)
            attention_masks = torch.stack(attention_masks, dim=1)
            
            return region_features, attention_masks


    class GraphAttentionLayer(nn.Module):
        """Single Graph Attention Layer"""
        def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
            super(GraphAttentionLayer, self).__init__()
            self.num_heads = num_heads
            self.out_features = out_features
            self.head_dim = out_features // num_heads
            
            assert out_features % num_heads == 0
            
            self.W_q = nn.Linear(in_features, out_features)
            self.W_k = nn.Linear(in_features, out_features)
            self.W_v = nn.Linear(in_features, out_features)
            self.W_o = nn.Linear(out_features, out_features)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(out_features)
            
        def forward(self, x, adjacency_matrix=None):
            B, N, _ = x.shape
            
            Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            
            if adjacency_matrix is not None:
                # Ensure adjacency matrix broadcasts correctly to (B, num_heads, N, N)
                # adjacency_matrix is (N, N), so it broadcasts automatically
                scores = scores.masked_fill(adjacency_matrix == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(B, N, self.out_features)
            
            output = self.W_o(context)
            output = self.dropout(output)
            
            return output, attn_weights


    class CrossRegionRelationalReasoning(nn.Module):
        """Graph Attention Network for reasoning across anatomical regions"""
        def __init__(self, feature_dim, hidden_dim=256, num_heads=4, num_layers=3, num_regions=5):
            super(CrossRegionRelationalReasoning, self).__init__()
            self.num_regions = num_regions
            
            self.region_to_node = nn.Linear(feature_dim, hidden_dim)
            
            self.gat_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ])
            
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            
            self.register_buffer('adjacency', torch.ones(num_regions, num_regions))
            
        def forward(self, region_features):
            B, N, C, H, W = region_features.shape
            
            node_features = F.adaptive_avg_pool2d(region_features.view(B*N, C, H, W), 1).view(B, N, C)
            x = self.region_to_node(node_features)
            
            for gat, norm in zip(self.gat_layers, self.layer_norms):
                residual = x
                x, attn_weights = gat(x, self.adjacency)
                x = norm(x + residual)
            
            return x, attn_weights


    class WaveletTransform(nn.Module):
        """2D Haar Wavelet Transform (GPU-Optimized)"""
        def __init__(self, wavelet='db1'):
            super(WaveletTransform, self).__init__()
            # Haar filters
            ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
            lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
            hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
            hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
            
            filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
            self.register_buffer('filters', filters)
        
        def forward(self, x):
            B, C, H, W = x.shape
            if H % 2 != 0 or W % 2 != 0:
                x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
            
            filters = self.filters.repeat(C, 1, 1, 1)
            output = F.conv2d(x, filters, stride=2, groups=C)
            
            B, _, H_out, W_out = output.shape
            return output.view(B, C, 4, H_out, W_out)


    class FrequencyInflammationDetector(nn.Module):
        """Detects inflammation patterns using frequency-domain analysis"""
        def __init__(self, in_channels):
            super(FrequencyInflammationDetector, self).__init__()
            
            self.rgb_to_hsv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(16, 3, kernel_size=1)
            )
            
            self.wavelet = WaveletTransform(wavelet='db1')
            
            self.low_freq_processor = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            
            self.high_freq_processor = nn.Sequential(
                nn.Conv2d(in_channels * 3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            
            self.fusion = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
        
        def forward(self, x):
            hsv_approx = self.rgb_to_hsv(x)
            coeffs = self.wavelet(hsv_approx)
            B, C, _, H, W = coeffs.shape
            
            low_freq = coeffs[:, :, 0, :, :]
            high_freq = coeffs[:, :, 1:, :, :].reshape(B, C*3, H, W)
            
            low_features = self.low_freq_processor(low_freq)
            high_features = self.high_freq_processor(high_freq)
            
            if low_features.shape[-2:] != high_features.shape[-2:]:
                low_features = F.interpolate(low_features, size=high_features.shape[-2:], 
                                            mode='bilinear', align_corners=False)
            
            inflammation_features = self.fusion(torch.cat([low_features, high_features], dim=1))
            
            return inflammation_features


    class SymptomGuidedFusion(nn.Module):
        """Adaptively fuses features from different sources"""
        def __init__(self, anatomical_dim, reasoning_dim, inflammation_dim, global_dim):
            super(SymptomGuidedFusion, self).__init__()
            
            total_dim = anatomical_dim + reasoning_dim + inflammation_dim + global_dim
            
            self.attention_gate = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2),
                nn.ReLU(True),
                nn.Linear(total_dim // 2, 4),
                nn.Softmax(dim=1)
            )
            
            self.anatomical_proj = nn.Linear(anatomical_dim, 256)
            self.reasoning_proj = nn.Linear(reasoning_dim, 256)
            self.inflammation_proj = nn.Linear(inflammation_dim, 256)
            self.global_proj = nn.Linear(global_dim, 256)
            
            self.fusion = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(256, 256)
            )
            
        def forward(self, anatomical_feat, reasoning_feat, inflammation_feat, global_feat):
            all_features = torch.cat([anatomical_feat, reasoning_feat, 
                                     inflammation_feat, global_feat], dim=1)
            
            weights = self.attention_gate(all_features)
            
            anatomical_proj = self.anatomical_proj(anatomical_feat)
            reasoning_proj = self.reasoning_proj(reasoning_feat)
            inflammation_proj = self.inflammation_proj(inflammation_feat)
            global_proj = self.global_proj(global_feat)
            
            fused = (weights[:, 0:1] * anatomical_proj + 
                    weights[:, 1:2] * reasoning_proj + 
                    weights[:, 2:3] * inflammation_proj + 
                    weights[:, 3:4] * global_proj)
            
            output = self.fusion(fused)
            
            return output, weights


    class MSAN(nn.Module):
        """Multi-Scale Anatomical Attention Network"""
        def __init__(self, num_regions=5, backbone='efficientnet_b3', pretrained=True):
            super(MSAN, self).__init__()
            self.num_regions = num_regions
            self.backbone_name = backbone
            
            if backbone == 'efficientnet_b3':
                efficientnet = models.efficientnet_b3(pretrained=pretrained)
                self.backbone = efficientnet.features
                backbone_out_channels = 1536
            elif backbone == 'resnet50':
                resnet = models.resnet50(pretrained=pretrained)
                self.backbone = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                )
                backbone_out_channels = 2048
            elif backbone == 'alexnet':
                alexnet = models.alexnet(pretrained=pretrained)
                self.backbone = alexnet.features
                self.backbone_adapter = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 1536, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                backbone_out_channels = 1536
            elif backbone == 'mobilenet_v2':
                mobilenet = models.mobilenet_v2(pretrained=pretrained)
                self.backbone = mobilenet.features
                self.backbone_adapter = nn.Sequential(
                    nn.Conv2d(1280, 1536, kernel_size=1),
                    nn.BatchNorm2d(1536),
                    nn.ReLU6(inplace=True)
                )
                backbone_out_channels = 1536
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            self.arpm = AnatomicalRegionProposal(in_channels=backbone_out_channels, num_regions=num_regions)
            self.crrr = CrossRegionRelationalReasoning(
                feature_dim=backbone_out_channels, hidden_dim=256, 
                num_heads=4, num_layers=3, num_regions=num_regions
            )
            self.fdid = FrequencyInflammationDetector(in_channels=3)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            anatomical_dim = num_regions * 256
            reasoning_dim = 256
            inflammation_dim = 64 * 7 * 7
            global_dim = backbone_out_channels
            
            self.sgff = SymptomGuidedFusion(
                anatomical_dim=anatomical_dim, reasoning_dim=reasoning_dim,
                inflammation_dim=inflammation_dim, global_dim=global_dim
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )
            
        def forward(self, x):
            B = x.size(0)
            
            backbone_features = self.backbone(x)
            
            if hasattr(self, 'backbone_adapter'):
                backbone_features = self.backbone_adapter(backbone_features)
            
            region_features, attention_masks = self.arpm(backbone_features)
            reasoning_features, graph_attention = self.crrr(region_features)
            
            anatomical_feat = reasoning_features.view(B, -1)
            reasoning_feat = reasoning_features.mean(dim=1)
            
            inflammation_features = self.fdid(x)
            inflammation_features = F.adaptive_avg_pool2d(inflammation_features, (7, 7))
            inflammation_feat = inflammation_features.view(B, -1)
            
            global_feat = self.global_pool(backbone_features).view(B, -1)
            
            fused_features, fusion_weights = self.sgff(
                anatomical_feat, reasoning_feat, inflammation_feat, global_feat
            )
            
            logits = self.classifier(fused_features)
            
            return logits.squeeze(1)


# ==================== DATASET ====================
class BinaryPharyngitisDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_folder = str(row.iloc[0])
        label = int(row.iloc[1])

        if label not in [0, 1]:
            raise ValueError(f"Invalid label {label} at index {idx}. Must be 0 or 1")

        folder_path = os.path.join(self.data_dir, img_folder)
        
        if os.path.exists(folder_path):
            img_files = [f for f in os.listdir(folder_path) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                img_path = os.path.join(folder_path, img_files[0])
            else:
                raise FileNotFoundError(f"No image found in {folder_path}")
        else:
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ==================== DATA TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()
        
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            with amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return epoch_loss, metrics


def calculate_metrics(labels, preds, probs=None):
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }
    
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            metrics['auc'] = roc_auc_score(labels, probs)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        metrics['specificity'] = 0
        metrics['sensitivity'] = 0
    
    # Guard against any NaNs from edge cases
    metrics = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in metrics.items()}
    
    return metrics


def train_model_single_fold(model, train_loader, val_loader, fold_num, num_epochs=NUM_EPOCHS):
    """Train a single model for one stratified split"""
    print(f"\n  Training run {fold_num}...")
    
    # Calculate class weights for imbalanced data
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_auc = 0
    best_epoch = 0
    patience_counter = 0
    best_metrics = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update schedulers
        scheduler_plateau.step(val_metrics['auc'])
        scheduler_cosine.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    if best_metrics is None:
        best_metrics = val_metrics
    
    return model, history, best_auc, best_metrics, best_epoch


def evaluate_model(model, test_loader, device):
    """Evaluate a trained model on a held-out test set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            with amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return epoch_loss, metrics


def train_model_single_split(model_builder, df_valid, data_dir, model_name, val_size=VAL_SPLIT, num_epochs=NUM_EPOCHS):
    """Train a model using a single stratified train/val split"""
    print(f"\n{'='*70}")
    print(f"Training: {model_name} with stratified train/val split")
    print(f"{'='*70}")
    
    labels = df_valid.iloc[:, 1].values
    train_df, val_df = train_test_split(
        df_valid, test_size=val_size, stratify=labels, random_state=42
    )
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    train_dataset = BinaryPharyngitisDataset(train_df.reset_index(drop=True), data_dir, train_transform)
    val_dataset = BinaryPharyngitisDataset(val_df.reset_index(drop=True), data_dir, val_transform)
    
    is_msan = model_name.startswith('MSAN')
    batch_size = 4 if is_msan else BATCH_SIZE
    loader_generator = torch.Generator().manual_seed(SEED)
    common_loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': torch.cuda.is_available(),
        'worker_init_fn': seed_worker,
        'generator': loader_generator
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    
    model = model_builder().to(device)
    
    trained_model, history, best_auc, best_metrics, best_epoch = train_model_single_fold(
        model, train_loader, val_loader, fold_num=1, num_epochs=num_epochs
    )
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1-Score: {best_metrics['f1']:.4f}")
    print(f"  AUC: {best_metrics['auc']:.4f}")
    print(f"  Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.4f}")
    
    return trained_model, best_metrics, history, best_epoch


def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC
    axes[2].plot(history['val_auc'], label='Val AUC', marker='s', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title(f'{model_name} - Validation AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    filename = f"training_history_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved as '{filename}'")


# ==================== MAIN ====================
def main():
    print("="*70)
    print("MULTI-MODEL COMPARISON FOR PHARYNGITIS CLASSIFICATION")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    excel_file = 'excel_binary.xlsx'
    data_dir = 'data_image_pharyngitis_nature'
    
    df = pd.read_excel(excel_file)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df.iloc[:, 1].value_counts().to_dict()}")
    
    # Filter valid samples
    df_valid = df[df.iloc[:, 1].isin([0, 1])].reset_index(drop=True)
    print(f"Valid samples (correct labels): {len(df_valid)}")

    if df_valid.empty:
        raise RuntimeError("No samples with valid labels (0 or 1) found in the dataset.")

    # Drop rows whose image folders are missing to avoid runtime errors
    folder_col = df_valid.columns[0]
    df_valid = df_valid.copy()
    df_valid['folder_exists'] = df_valid[folder_col].astype(str).apply(
        lambda folder: os.path.isdir(os.path.join(data_dir, folder))
    )
    missing_count = (~df_valid['folder_exists']).sum()
    if missing_count:
        missing_folders = df_valid.loc[~df_valid['folder_exists'], folder_col].astype(str).tolist()
        print(f"Skipping {missing_count} entries with missing folders.")
        print(f"Example missing folders: {missing_folders[:5]}")
    
    # Filter and drop the helper column
    df_valid = df_valid[df_valid['folder_exists']].reset_index(drop=True)
    df_valid = df_valid.drop(columns=['folder_exists'])

    if df_valid.empty:
        raise RuntimeError("No samples left after removing entries with missing folders.")

    print(f"Samples with existing folders: {len(df_valid)}")
    print(f"Final class distribution: {df_valid.iloc[:, 1].value_counts().to_dict()}")
    
    print(f"\n2. Splitting data: 80% train+val, 20% test (stratified)")
    labels_all = df_valid.iloc[:, 1].values
    train_val_df, test_df = train_test_split(
        df_valid, test_size=TRAIN_TEST_SPLIT, stratify=labels_all, random_state=42
    )
    print(f"Train+Val samples: {len(train_val_df)}, Test samples: {len(test_df)}")
    print(f"Within Train+Val: using {int((1-VAL_SPLIT)*100)}% train / {int(VAL_SPLIT*100)}% val (stratified)")
    
    if test_df.empty:
        print("Warning: test split is empty after stratification.")
    
    # Define models to test
    models_to_test = [
        #Pretrained Baselines
        ('ResNet50 Baseline (Pretrained)', lambda: BaselineBinary('resnet50', pretrained=True)),
        ('AlexNet Baseline (Pretrained)', lambda: BaselineBinary('alexnet', pretrained=True)),
        ('MobileNetV2 Baseline (Pretrained)', lambda: BaselineBinary('mobilenet_v2', pretrained=True)),
        
        # Non-Pretrained Baselines
        ('ResNet50 Baseline (From Scratch)', lambda: BaselineBinary('resnet50', pretrained=False)),
        ('AlexNet Baseline (From Scratch)', lambda: BaselineBinary('alexnet', pretrained=False)),
        ('MobileNetV2 Baseline (From Scratch)', lambda: BaselineBinary('mobilenet_v2', pretrained=False)),
        
        # Pretrained Hybrid Models
        ('Hybrid FGCR+CSBA (ResNet50, Pretrained)', lambda: HybridFGCR_CSBA('resnet50', pretrained=True)),
        ('Hybrid FGCR+CSBA (AlexNet, Pretrained)', lambda: HybridFGCR_CSBA('alexnet', pretrained=True)),
        ('Hybrid FGCR+CSBA (MobileNetV2, Pretrained)', lambda: HybridFGCR_CSBA('mobilenet_v2', pretrained=True)),
        
        # Non-Pretrained Hybrid Models
        ('Hybrid FGCR+CSBA (ResNet50, From Scratch)', lambda: HybridFGCR_CSBA('resnet50', pretrained=False)),
        ('Hybrid FGCR+CSBA (AlexNet, From Scratch)', lambda: HybridFGCR_CSBA('alexnet', pretrained=False)),
        ('Hybrid FGCR+CSBA (MobileNetV2, From Scratch)', lambda: HybridFGCR_CSBA('mobilenet_v2', pretrained=False)),
    ]
    
    # Add MSAN models if pywt is available
    if PYWT_AVAILABLE:
        models_to_test.extend([
            # Pretrained MSAN Models
           ('MSAN (EfficientNet-B3, Pretrained)', lambda: MSAN(num_regions=5, backbone='efficientnet_b3', pretrained=True)),
             ('MSAN (ResNet50, Pretrained)', lambda: MSAN(num_regions=5, backbone='resnet50', pretrained=True)),
            ('MSAN (AlexNet, Pretrained)', lambda: MSAN(num_regions=5, backbone='alexnet', pretrained=True)),
            ('MSAN (MobileNetV2, Pretrained)', lambda: MSAN(num_regions=5, backbone='mobilenet_v2', pretrained=True)),
            
            # Non-Pretrained MSAN Models
            ('MSAN (EfficientNet-B3, From Scratch)', lambda: MSAN(num_regions=5, backbone='efficientnet_b3', pretrained=False)),
            ('MSAN (ResNet50, From Scratch)', lambda: MSAN(num_regions=5, backbone='resnet50', pretrained=False)),
            ('MSAN (AlexNet, From Scratch)', lambda: MSAN(num_regions=5, backbone='alexnet', pretrained=False)),
            ('MSAN (MobileNetV2, From Scratch)', lambda: MSAN(num_regions=5, backbone='mobilenet_v2', pretrained=False)),
         ])
        print(f"\n✓ PyWavelets detected. Added {8} MSAN models (4 pretrained + 4 from scratch) to testing queue.")
    else:
        print(f"\n✗ PyWavelets not available. Skipping MSAN models.")
        print("  Install with: pip install PyWavelets")
    
    print(f"\nTotal models to test: {len(models_to_test)}")
    
    results = []
    
    # Train each model with stratified split
    for model_name, model_builder in models_to_test:
        # Count parameters
        temp_model = model_builder()
        total_params = sum(p.numel() for p in temp_model.parameters())
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        del temp_model
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train with single stratified split
        trained_model, best_metrics, history, best_epoch = train_model_single_split(
            model_builder, train_val_df, data_dir, model_name, val_size=VAL_SPLIT, num_epochs=NUM_EPOCHS
        )
        
        # Optional test evaluation
        test_metrics = None
        test_status = "empty after split" if test_df is not None and len(test_df) == 0 else "no test split"
        if test_df is not None and len(test_df) > 0:
            is_msan = model_name.startswith('MSAN')
            batch_size = 4 if is_msan else BATCH_SIZE
            loader_generator = torch.Generator().manual_seed(SEED)
            common_loader_kwargs = {
                'batch_size': batch_size,
                'num_workers': 4,
                'pin_memory': torch.cuda.is_available(),
                'worker_init_fn': seed_worker,
                'generator': loader_generator
            }
            test_dataset = BinaryPharyngitisDataset(test_df, data_dir, val_transform)
            test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
            
            test_loss, test_metrics = evaluate_model(trained_model, test_loader, device)
            test_status = "ok"
        elif test_df is not None and len(test_df) == 0:
            print("Skipping test evaluation (no test samples after filtering).")
            test_status = "empty after filtering"
        
        # If we did test evaluation, print
        if test_metrics is not None:
            print(f"\nTest Results:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  F1-Score: {test_metrics['f1']:.4f}")
            print(f"  AUC: {test_metrics['auc']:.4f}")
            print(f"  Specificity: {test_metrics['specificity']:.4f}")
            print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        
        # Save results
        results.append({
            'Model': model_name,
            'Parameters': total_params,
            'Val Accuracy': f"{best_metrics['accuracy']:.4f}",
            'Val Precision': f"{best_metrics['precision']:.4f}",
            'Val Recall': f"{best_metrics['recall']:.4f}",
            'Val F1': f"{best_metrics['f1']:.4f}",
            'Val AUC': f"{best_metrics['auc']:.4f}",
            'Val Specificity': f"{best_metrics['specificity']:.4f}",
            'Val Sensitivity': f"{best_metrics['sensitivity']:.4f}",
            'Best Epoch': best_epoch,
            'AUC Raw': best_metrics['auc'],  # For sorting
            'Test Accuracy': f"{test_metrics['accuracy']:.4f}" if test_metrics else 'N/A',
            'Test Precision': f"{test_metrics['precision']:.4f}" if test_metrics else 'N/A',
            'Test Recall': f"{test_metrics['recall']:.4f}" if test_metrics else 'N/A',
            'Test F1': f"{test_metrics['f1']:.4f}" if test_metrics else 'N/A',
            'Test AUC': f"{test_metrics['auc']:.4f}" if test_metrics else 'N/A',
            'Test Specificity': f"{test_metrics['specificity']:.4f}" if test_metrics else 'N/A',
            'Test Sensitivity': f"{test_metrics['sensitivity']:.4f}" if test_metrics else 'N/A',
            'Test Status': test_status,
        })

        # Plot training history
        plot_training_history(history, f"{model_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Create results table
    print(f"\n{'='*70}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_excel('model_comparison_results.xlsx', index=False)
    print(f"\nResults saved to 'model_comparison_results.xlsx'")
    
    # Find best model
    best_model_idx = results_df['AUC Raw'].idxmax()
    best_model = results_df.loc[best_model_idx]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model['Model']}")
    print(f"Val AUC: {best_model['Val AUC']}")
    print(f"Val Accuracy: {best_model['Val Accuracy']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    set_global_seed(SEED)
    main()
