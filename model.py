"""
Multi-Scale Anatomical Attention Network (MSAN)
For Bacterial vs Non-bacterial Pharyngitis Classification

Novel Architecture Components:
1. Anatomical Region Proposal Module (ARPM)
2. Cross-Region Relational Reasoning (CRRR) with Graph Attention
3. Frequency-Domain Inflammation Detector (FDID)
4. Symptom-Guided Feature Fusion (SGFF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pywt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 16
NUM_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 0.0001
IMG_SIZE = 224
NUM_CLASSES = 2
NUM_REGIONS = 5  # Left tonsil, Right tonsil, Uvula, Posterior pharyngeal wall, Global

# ==================== 1. ANATOMICAL REGION PROPOSAL MODULE ====================
class SpatialTransformerNetwork(nn.Module):
    """Learns to focus on specific anatomical regions"""
    def __init__(self, in_channels):
        super(SpatialTransformerNetwork, self).__init__()
        # Localization network
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
        
        # Calculate the size after convolutions
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x, theta


class AnatomicalRegionProposal(nn.Module):
    """
    Proposes attention masks for 5 anatomical regions:
    1. Left tonsil
    2. Right tonsil
    3. Uvula
    4. Posterior pharyngeal wall
    5. Global context
    """
    def __init__(self, in_channels, num_regions=5):
        super(AnatomicalRegionProposal, self).__init__()
        self.num_regions = num_regions
        
        # Spatial transformers for each region (except global)
        self.region_transformers = nn.ModuleList([
            SpatialTransformerNetwork(in_channels) 
            for _ in range(num_regions - 1)
        ])
        
        # Region-specific feature extractors
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
        
        # Attention mask generators
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
        
        # Process each region
        for i in range(self.num_regions):
            if i < self.num_regions - 1:
                # Apply spatial transformer for specific anatomical regions
                transformed_x, theta = self.region_transformers[i](x)
                region_feat = self.region_extractors[i](transformed_x)
            else:
                # Global context (no transformation)
                region_feat = self.region_extractors[i](x)
            
            # Generate attention mask
            mask = self.mask_generators[i](region_feat)
            
            # Apply mask
            masked_feat = region_feat * mask
            
            region_features.append(masked_feat)
            attention_masks.append(mask)
        
        # Stack region features: [B, num_regions, C, H, W]
        region_features = torch.stack(region_features, dim=1)
        attention_masks = torch.stack(attention_masks, dim=1)
        
        return region_features, attention_masks


# ==================== 2. GRAPH ATTENTION FOR CROSS-REGION REASONING ====================
class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer"""
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        # Multi-head attention parameters
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
        self.W_o = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x, adjacency_matrix=None):
        # x: [B, num_nodes, in_features]
        B, N, _ = x.shape
        
        # Multi-head attention
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply adjacency matrix if provided
        if adjacency_matrix is not None:
            scores = scores.masked_fill(adjacency_matrix.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.out_features)
        
        # Output projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output, attn_weights


class CrossRegionRelationalReasoning(nn.Module):
    """
    Graph Attention Network for reasoning across anatomical regions
    """
    def __init__(self, feature_dim, hidden_dim=256, num_heads=4, num_layers=3, num_regions=5):
        super(CrossRegionRelationalReasoning, self).__init__()
        self.num_regions = num_regions
        
        # Project region features to graph node features
        self.region_to_node = nn.Linear(feature_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Learnable adjacency matrix (fully connected initially)
        self.register_buffer('adjacency', torch.ones(num_regions, num_regions))
        
    def forward(self, region_features):
        # region_features: [B, num_regions, C, H, W]
        B, N, C, H, W = region_features.shape
        
        # Global average pooling for each region
        node_features = F.adaptive_avg_pool2d(region_features.view(B*N, C, H, W), 1).view(B, N, C)
        
        # Project to graph node space
        x = self.region_to_node(node_features)  # [B, num_regions, hidden_dim]
        
        # Apply graph attention layers
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x, attn_weights = gat(x, self.adjacency)
            x = norm(x + residual)
        
        return x, attn_weights  # [B, num_regions, hidden_dim], attention weights


# ==================== 3. FREQUENCY-DOMAIN INFLAMMATION DETECTOR ====================
class WaveletTransform(nn.Module):
    """2D Discrete Wavelet Transform"""
    def __init__(self, wavelet='db1'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply DWT to each channel
        coeffs_list = []
        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                img = x[b, c].cpu().detach().numpy()
                coeffs = pywt.dwt2(img, self.wavelet)
                cA, (cH, cV, cD) = coeffs
                
                # Stack all coefficients
                all_coeffs = np.stack([cA, cH, cV, cD], axis=0)
                batch_coeffs.append(all_coeffs)
            
            batch_coeffs = np.stack(batch_coeffs, axis=0)
            coeffs_list.append(batch_coeffs)
        
        coeffs_array = np.stack(coeffs_list, axis=0)
        return torch.from_numpy(coeffs_array).float().to(x.device)


class FrequencyInflammationDetector(nn.Module):
    """
    Detects inflammation patterns using frequency-domain analysis
    """
    def __init__(self, in_channels):
        super(FrequencyInflammationDetector, self).__init__()
        
        # Color space conversion layers
        self.rgb_to_hsv = self._create_color_converter()
        
        # Wavelet transform
        self.wavelet = WaveletTransform(wavelet='db1')
        
        # Frequency band processors
        # After DWT: 4 subbands (LL, LH, HL, HH) for each input channel
        self.low_freq_processor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.high_freq_processor = nn.Sequential(
            nn.Conv2d(in_channels * 3, 64, kernel_size=3, padding=1),  # 3 high-freq subbands
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # Inflammation feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
    def _create_color_converter(self):
        # Simplified RGB to HSV conversion approximation using convolution
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=1)
        )
    
    def forward(self, x):
        # x: [B, 3, H, W] RGB image
        
        # Color space conversion (simplified)
        hsv_approx = self.rgb_to_hsv(x)
        
        # Apply wavelet transform
        coeffs = self.wavelet(hsv_approx)  # [B, C, 4, H/2, W/2]
        B, C, _, H, W = coeffs.shape
        
        # Separate low and high frequency components
        low_freq = coeffs[:, :, 0, :, :]  # LL subband: [B, C, H/2, W/2]
        high_freq = coeffs[:, :, 1:, :, :].reshape(B, C*3, H, W)  # LH, HL, HH
        
        # Process frequency bands
        low_features = self.low_freq_processor(low_freq)
        high_features = self.high_freq_processor(high_freq)
        
        # Upsample to match dimensions
        if low_features.shape[-2:] != high_features.shape[-2:]:
            low_features = F.interpolate(low_features, size=high_features.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse features
        inflammation_features = self.fusion(torch.cat([low_features, high_features], dim=1))
        
        return inflammation_features


# ==================== 4. SYMPTOM-GUIDED FEATURE FUSION ====================
class SymptomGuidedFusion(nn.Module):
    """
    Adaptively fuses features from different sources based on learned importance
    """
    def __init__(self, anatomical_dim, reasoning_dim, inflammation_dim, global_dim):
        super(SymptomGuidedFusion, self).__init__()
        
        total_dim = anatomical_dim + reasoning_dim + inflammation_dim + global_dim
        
        # Attention gate for adaptive fusion
        self.attention_gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(True),
            nn.Linear(total_dim // 2, 4),  # 4 weights for 4 feature sources
            nn.Softmax(dim=1)
        )
        
        # Feature projections to common dimension
        self.anatomical_proj = nn.Linear(anatomical_dim, 256)
        self.reasoning_proj = nn.Linear(reasoning_dim, 256)
        self.inflammation_proj = nn.Linear(inflammation_dim, 256)
        self.global_proj = nn.Linear(global_dim, 256)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        
    def forward(self, anatomical_feat, reasoning_feat, inflammation_feat, global_feat):
        # Concatenate all features for attention computation
        all_features = torch.cat([anatomical_feat, reasoning_feat, 
                                 inflammation_feat, global_feat], dim=1)
        
        # Compute attention weights
        weights = self.attention_gate(all_features)  # [B, 4]
        
        # Project features to common space
        anatomical_proj = self.anatomical_proj(anatomical_feat)
        reasoning_proj = self.reasoning_proj(reasoning_feat)
        inflammation_proj = self.inflammation_proj(inflammation_feat)
        global_proj = self.global_proj(global_feat)
        
        # Weighted fusion
        fused = (weights[:, 0:1] * anatomical_proj + 
                weights[:, 1:2] * reasoning_proj + 
                weights[:, 2:3] * inflammation_proj + 
                weights[:, 3:4] * global_proj)
        
        # Final processing
        output = self.fusion(fused)
        
        return output, weights


# ==================== COMPLETE MSAN MODEL ====================
class MSAN(nn.Module):
    """
    Multi-Scale Anatomical Attention Network
    Complete architecture for pharyngitis classification
    Supports multiple backbones: efficientnet_b3, resnet50, alexnet, mobilenet_v2
    """
    def __init__(self, num_classes=2, num_regions=5, backbone='efficientnet_b3', pretrained=True):
        super(MSAN, self).__init__()
        self.num_regions = num_regions
        self.backbone_name = backbone
        
        # Initialize backbone based on selection
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
            # Add adapter to increase channels
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
            # Add adapter to match expected channels
            self.backbone_adapter = nn.Sequential(
                nn.Conv2d(1280, 1536, kernel_size=1),
                nn.BatchNorm2d(1536),
                nn.ReLU6(inplace=True)
            )
            backbone_out_channels = 1536
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from: efficientnet_b3, resnet50, alexnet, mobilenet_v2")
        
        # 1. Anatomical Region Proposal Module
        self.arpm = AnatomicalRegionProposal(
            in_channels=backbone_out_channels,
            num_regions=num_regions
        )
        
        # 2. Cross-Region Relational Reasoning
        self.crrr = CrossRegionRelationalReasoning(
            feature_dim=backbone_out_channels,
            hidden_dim=256,
            num_heads=4,
            num_layers=3,
            num_regions=num_regions
        )
        
        # 3. Frequency-Domain Inflammation Detector
        # Apply to input image
        self.fdid = FrequencyInflammationDetector(in_channels=3)
        
        # 4. Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate dimensions for fusion
        anatomical_dim = num_regions * 256  # From CRRR output
        reasoning_dim = 256  # From graph reasoning
        inflammation_dim = 64 * 7 * 7  # From FDID output
        global_dim = backbone_out_channels  # From backbone
        
        # 5. Symptom-Guided Feature Fusion
        self.sgff = SymptomGuidedFusion(
            anatomical_dim=anatomical_dim,
            reasoning_dim=reasoning_dim,
            inflammation_dim=inflammation_dim,
            global_dim=global_dim
        )
        
        # 6. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        B = x.size(0)
        
        # Extract backbone features
        backbone_features = self.backbone(x)  # [B, backbone_out_channels, H', W']
        
        # Apply adapter if using alexnet or mobilenet_v2
        if hasattr(self, 'backbone_adapter'):
            backbone_features = self.backbone_adapter(backbone_features)
        
        # 1. Anatomical Region Proposal
        region_features, attention_masks = self.arpm(backbone_features)
        # region_features: [B, num_regions, C, H', W']
        
        # 2. Cross-Region Relational Reasoning
        reasoning_features, graph_attention = self.crrr(region_features)
        # reasoning_features: [B, num_regions, 256]
        
        # Flatten anatomical features
        anatomical_feat = reasoning_features.view(B, -1)
        
        # Global reasoning (average across regions)
        reasoning_feat = reasoning_features.mean(dim=1)
        
        # 3. Frequency-Domain Inflammation Detection
        inflammation_features = self.fdid(x)  # [B, 64, H'', W'']
        inflammation_feat = inflammation_features.view(B, -1)
        
        # 4. Global features
        global_feat = self.global_pool(backbone_features).view(B, -1)
        
        # 5. Symptom-Guided Fusion
        fused_features, fusion_weights = self.sgff(
            anatomical_feat,
            reasoning_feat,
            inflammation_feat,
            global_feat
        )
        
        # 6. Classification
        logits = self.classifier(fused_features)
        
        return logits, {
            'attention_masks': attention_masks,
            'graph_attention': graph_attention,
            'fusion_weights': fusion_weights
        }


# ==================== DATASET ====================
class PharyngitisDataset(Dataset):
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
            raise ValueError(f"Invalid label {label}")

        folder_path = os.path.join(self.data_dir, img_folder)
        
        if os.path.exists(folder_path):
            img_files = [f for f in os.listdir(folder_path) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                img_path = os.path.join(folder_path, img_files[0])
            else:
                raise FileNotFoundError(f"No image in {folder_path}")
        else:
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ==================== DATA TRANSFORMS ====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, aux_outputs = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
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
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits, aux_outputs = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
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
        metrics['auc'] = roc_auc_score(labels, probs)
    else:
        metrics['auc'] = 0.0
    
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


# ==================== MAIN TRAINING ====================
def main():
    print("="*70)
    print("MULTI-SCALE ANATOMICAL ATTENTION NETWORK (MSAN)")
    print("For Pharyngitis Classification")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    excel_file = 'excel_binary.xlsx'
    data_dir = 'data_image_pharyngitis_nature'
    
    df = pd.read_excel(excel_file)
    print(f"   Total samples: {len(df)}")
    
    # Filter and validate
    df_valid = df[df.iloc[:, 1].isin([0, 1])].reset_index(drop=True)
    folder_col = df_valid.columns[0]
    df_valid = df_valid.copy()
    df_valid['folder_exists'] = df_valid[folder_col].astype(str).apply(
        lambda folder: os.path.isdir(os.path.join(data_dir, folder))
    )
    df_valid = df_valid[df_valid['folder_exists']].drop(columns=['folder_exists']).reset_index(drop=True)
    
    print(f"   Valid samples: {len(df_valid)}")
    print(f"   Class distribution: {df_valid.iloc[:, 1].value_counts().to_dict()}")
    
    # Split data
    print("\n2. Splitting data...")
    labels = df_valid.iloc[:, 1]
    train_val_df, test_df = train_test_split(df_valid, test_size=0.2, random_state=42, stratify=labels)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, 
                                        stratify=train_val_df.iloc[:, 1])
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = PharyngitisDataset(train_df, data_dir, train_transform)
    val_dataset = PharyngitisDataset(val_df, data_dir, val_transform)
    test_dataset = PharyngitisDataset(test_df, data_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    print("\n3. Initializing MSAN model...")
    model = MSAN(num_classes=NUM_CLASSES, num_regions=NUM_REGIONS, backbone='efficientnet_b3', pretrained=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")