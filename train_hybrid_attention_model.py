"""
Hybrid Attention Model for Binary Pharyngitis Classification
ResNet50 backbone + Frequency-Gated Channel Recalibration + Cross-Scale Bi-Attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

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

# ==================== CBAM COMPONENTS (FOR BASELINE) ====================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ==================== FREQUENCY-GATED CHANNEL RECALIBRATION ====================
class FrequencyGatedChannelRecalibration(nn.Module):
    """
    Applies spectral band gating to recalibrate channels and returns
    both the recalibrated feature map and low/high-frequency energy tokens.
    """
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
        yy, xx = torch.meshgrid(y, x)
        radial = torch.sqrt(xx ** 2 + yy ** 2)
        mask_low = (radial <= self.cutoff_ratio).float()
        mask_high = torch.ones_like(mask_low) - mask_low
        return mask_low, mask_high


# ==================== CROSS-SCALE BI-DIRECTIONAL ATTENTION ====================
class CrossScaleBiAttention(nn.Module):
    """
    Couples mid- and high-resolution ResNet tokens via bi-directional attention
    and emits both fused high-resolution features and a compact cross-scale token.
    """
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


# ==================== BASELINE BACKBONES ====================
class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Binary, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


class AlexNetBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetBinary, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


class MobileNetV2Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Binary, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


class CBAMResNet50(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super(CBAMResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.cbam1, self.cbam2, self.cbam3, self.cbam4, self.classifier]:
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


# ==================== HYBRID ATTENTION MODEL ====================
class HybridAttentionResNet(nn.Module):
    """
    ResNet50 backbone augmented with:
    - Frequency-Gated Channel Recalibration (FGCR) on deeper residual stages
    - Cross-Scale Bi-Attention (CSBA) between layer3 and layer4 tokens
    - Spectral-energy aware classifier head
    """
    def __init__(self, pretrained=True, num_heads=4, attn_dim=128, attn_dropout=0.2, cls_dropout=0.5):
        super(HybridAttentionResNet, self).__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.fgcr3 = FrequencyGatedChannelRecalibration(1024)
        self.fgcr4 = FrequencyGatedChannelRecalibration(2048)
        
        self.cross_scale = CrossScaleBiAttention(
            c_low=1024,
            c_high=2048,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        classifier_in = 2048 + self.cross_scale.token_dim + 2
        self.classifier = nn.Sequential(
            nn.Dropout(cls_dropout),
            nn.Linear(classifier_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(512, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        modules = [
            self.fgcr3, self.fgcr4,
            self.cross_scale, self.classifier
        ]
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
        
        x, bridge_token = self.cross_scale(low_scale, x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        stats = torch.cat([spectral_token, bridge_token], dim=1)
        logits = self.classifier(torch.cat([x, stats], dim=1))
        
        return logits.squeeze(1)


# ==================== MODEL BUILDERS ====================
MODEL_BUILDERS = {
    'resnet50': lambda: ResNet50Binary(pretrained=True),
    'alexnet': lambda: AlexNetBinary(pretrained=True),
    'mobilenet_v2': lambda: MobileNetV2Binary(pretrained=True),
    'cbam_resnet50': lambda: CBAMResNet50(pretrained=True, dropout=0.3),
    'hybrid_fgcr_csba': lambda: HybridAttentionResNet(pretrained=True, num_heads=4, attn_dim=128, attn_dropout=0.2, cls_dropout=0.5),
}

MODEL_DISPLAY_NAMES = {
    'resnet50': 'ResNet-50',
    'alexnet': 'AlexNet',
    'mobilenet_v2': 'MobileNetV2',
    'cbam_resnet50': 'CBAM-ResNet50',
    'hybrid_fgcr_csba': 'Hybrid FGCR+CSBA',
}


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
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
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
        metrics['auc'] = roc_auc_score(labels, probs)
    else:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def train_single_model(model_key, train_loader, val_loader, test_loader, pos_weight):
    display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    print("\n" + "="*70)
    print(f"TRAINING MODEL: {display_name}")
    print("="*70)
    
    if model_key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model = MODEL_BUILDERS[model_key]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_metrics = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience = PATIENCE
    patience_counter = 0
    checkpoint_path = f'best_{model_key}.pth'
    history_plot = f'{model_key}_training_history.png'
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nModel {display_name} - Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"Val   Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"Val   Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        print(f"Val   F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_loss
            best_metrics = {k: float(v) for k, v in val_metrics.items()}
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_metrics['accuracy'],
                'val_metrics': val_metrics,
            }, checkpoint_path)
            
            print(f"   * New best model saved for {display_name}! (Val Acc: {val_metrics['accuracy']*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n   Early stopping triggered for {display_name} (no improvement for {patience} epochs)")
                break
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{display_name} Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([x*100 for x in history['train_acc']], label='Train Acc', linewidth=2)
    axes[1].plot([x*100 for x in history['val_acc']], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{display_name} Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(history_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   * Training history saved as '{history_plot}'")
    
    # Load best weights for final evaluation
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\nTest Results for {display_name}:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"   Test Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print(f"   Test Sensitivity: {test_metrics['sensitivity']:.4f}, Specificity: {test_metrics['specificity']:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    summary = {
        'model_key': model_key,
        'model_name': display_name,
        'best_val_accuracy': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'checkpoint': checkpoint_path,
        'history_plot': history_plot,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_metrics['accuracy']),
    }
    if best_metrics is not None:
        for metric_name, metric_value in best_metrics.items():
            summary[f'val_{metric_name}'] = metric_value
    for metric_name, metric_value in test_metrics.items():
        summary[f'test_{metric_name}'] = float(metric_value)
    
    return summary


# ==================== MAIN TRAINING ====================
def main():
    print("="*70)
    print("HYBRID ATTENTION MODEL FOR BINARY PHARYNGITIS CLASSIFICATION")
    print("ResNet50 + FGCR + Cross-Scale Bi-Attention")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    excel_file = 'excel_binary.xlsx'
    data_dir = 'data_image_pharyngitis_nature'
    
    df = pd.read_excel(excel_file)
    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution: {df.iloc[:, 1].value_counts().to_dict()}")
    
    # Filter valid samples
    df_valid = df[df.iloc[:, 1].isin([0, 1])].reset_index(drop=True)
    print(f"   Valid samples (correct labels): {len(df_valid)}")

    # Drop rows whose image folders are missing to avoid runtime errors
    folder_col = df_valid.columns[0]
    df_valid = df_valid.copy()
    df_valid['folder_exists'] = df_valid[folder_col].astype(str).apply(
        lambda folder: os.path.isdir(os.path.join(data_dir, folder))
    )
    missing_count = (~df_valid['folder_exists']).sum()
    if missing_count:
        missing_folders = df_valid.loc[~df_valid['folder_exists'], folder_col].astype(str).tolist()
        print(f"   Skipping {missing_count} entries with missing folders.")
        print(f"   Example missing folders: {missing_folders[:5]}")
    df_valid = df_valid[df_valid['folder_exists']].drop(columns=['folder_exists']).reset_index(drop=True)

    if df_valid.empty:
        raise RuntimeError("No samples left after removing entries with missing folders.")

    print(f"   Samples with existing folders: {len(df_valid)}")
    
    # Calculate class weights
    class_counts = df_valid.iloc[:, 1].value_counts()
    if len(class_counts) == 2:
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
        print(f"   Positive class weight: {pos_weight.item():.3f}")
    else:
        pos_weight = None
    
    # Split data
    print("\n2. Splitting data (80% train, 10% val, 20% test)...")
    from sklearn.model_selection import train_test_split
    
    labels = df_valid.iloc[:, 1]
    can_stratify = labels.nunique() == 2 and labels.value_counts().min() >= 2
    stratify_labels = labels if can_stratify else None
    if not can_stratify:
        print("   WARNING: Stratified split disabled due to insufficient samples per class.")
    
    train_val_df, test_df = train_test_split(
        df_valid,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels
    )
    
    train_val_labels = train_val_df.iloc[:, 1]
    can_stratify_tv = train_val_labels.nunique() == 2 and train_val_labels.value_counts().min() >= 2
    stratify_tv = train_val_labels if can_stratify_tv else None
    if not can_stratify_tv:
        print("   WARNING: Stratified split disabled when creating validation set.")
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,
        random_state=42,
        stratify=stratify_tv
    )
    
    print(f"   Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets and loaders
    train_dataset = BinaryPharyngitisDataset(train_df, data_dir, train_transform)
    val_dataset = BinaryPharyngitisDataset(val_df, data_dir, val_transform)
    test_dataset = BinaryPharyngitisDataset(test_df, data_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    print("\n3. Training model suite sequentially...")
    model_order = ['resnet50', 'alexnet', 'mobilenet_v2', 'cbam_resnet50', 'hybrid_fgcr_csba']
    results = []
    
    for key in model_order:
        summary = train_single_model(key, train_loader, val_loader, test_loader, pos_weight)
        results.append(summary)
    
    results_df = pd.DataFrame(results)
    results_path = 'model_suite_results.xlsx'
    results_df.to_excel(results_path, index=False)
    
    print("\n" + "="*70)
    print("MODEL SUITE TRAINING COMPLETED")
    print("="*70)
    print(results_df[['model_name', 'best_val_accuracy', 'best_val_loss']])
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
