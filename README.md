# Hybrid Attention Model for Pharyngitis Classification

Deep learning framework for binary pharyngitis classification (Bacterial vs Non-Bacterial) using hybrid attention mechanisms.

## 🔬 Overview

This project implements a hybrid attention model combining:
- **ResNet50** backbone for feature extraction
- **Frequency-Gated Channel Recalibration (FGCR)** for spectral analysis
- **Cross-Scale Bi-Attention (CSBA)** for multi-scale feature fusion

The model achieves high performance on pharyngitis image classification by leveraging both spatial and frequency domain information.

## 🏗️ Architecture

For a detailed explanation of the architecture and visual schematic, see:
- **[ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md)** - Complete technical documentation
- **[model_architecture_schematic.png](model_architecture_schematic.png)** - Visual diagram of the model

### Components

1. **Frequency-Gated Channel Recalibration (FGCR)**
   - Applies FFT-based spectral analysis
   - Separates low and high-frequency components
   - Generates frequency-aware channel weights

2. **Cross-Scale Bi-Attention (CSBA)**
   - Multi-head self-attention mechanism
   - Processes low and high-frequency tokens separately
   - Bi-directional feature fusion

3. **Hybrid Attention Integration**
   - Combines CBAM (Convolutional Block Attention Module)
   - Frequency-domain channel recalibration
   - Cross-scale spatial attention

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
Pillow>=9.3.0
tqdm>=4.64.0
matplotlib>=3.6.0
openpyxl>=3.0.0
```

Install dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn Pillow tqdm matplotlib openpyxl
```

## 📁 Project Structure

```
phd/
├── train_hybrid_attention_model.py    # Main training script
├── best_hybrid_attention_model.pth    # Best model checkpoint
├── best_hybrid_fgcr_csba.pth         # Alternative checkpoint
├── hybrid_attention_training_history.png  # Training curves
├── hybrid_fgcr_csba_training_history.png  # Alternative training curves
├── excel.xlsx                         # Dataset labels and metadata
├── data_image_pharyngitis_nature/     # Image dataset directory
└── basepharyngitis-main/              # Reference implementation
```

## 🚀 Usage

### Training

1. **Prepare your dataset:**
   - Place images in `data_image_pharyngitis_nature/` directory
   - Ensure `excel.xlsx` contains image filenames and binary labels (0 or 1)
   - Excel format: Column 0 = image filename, Column 1 = label

2. **Configure hyperparameters** in `train_hybrid_attention_model.py`:
```python
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
IMG_SIZE = 224
```

3. **Run training:**
```bash
python train_hybrid_attention_model.py
```

### Model Selection

The script supports multiple model architectures:
- CBAM-ResNet50 (baseline)
- Hybrid Attention Model (FGCR + CBAM)
- Hybrid FGCR-CSBA Model (full architecture)

Select by modifying the `model_choice` variable in the main function.

## 📊 Features

- **Binary Classification**: Bacterial (1) vs Non-Bacterial (0)
- **Data Augmentation**: Rotation, flipping, color jittering, Gaussian blur
- **Class Imbalance Handling**: Weighted BCEWithLogitsLoss
- **Advanced Scheduling**: ReduceLROnPlateau + CosineAnnealingWarmRestarts
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC, Specificity, Sensitivity
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Visualization**: Automatic training history plots

## 🎯 Performance Metrics

The model evaluates using:
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC
- Specificity & Sensitivity
- Confusion Matrix

## 📈 Training History

Training curves are automatically saved showing:
- Training and validation loss
- Training and validation accuracy
- Learning rate progression

## 💾 Model Checkpoints

Best models are automatically saved based on validation AUC score:
- `best_hybrid_attention_model.pth` - Complete model state
- Includes: model weights, optimizer state, scheduler state, epoch, metrics

## 🔧 Advanced Configuration

### Attention Parameters
```python
# FGCR settings
reduction=8          # Channel reduction ratio
cutoff_ratio=0.25    # Frequency cutoff threshold

# CSBA settings
num_heads=8          # Multi-head attention
dropout=0.1          # Attention dropout rate
```

### Training Parameters
```python
PATIENCE = 7         # Early stopping patience
weight_decay=1e-4    # L2 regularization
pos_weight           # Automatic class balancing
```

## 📝 Dataset Format

Excel file (`excel.xlsx`) should contain:
- Column 0: Image filename (e.g., "001.jpg")
- Column 1: Binary label (0 or 1)
- 0 = Non-Bacterial
- 1 = Bacterial

Images should be in `data_image_pharyngitis_nature/` subdirectories.

## 🔬 Reference

Based on pharyngitis classification research implementing frequency-domain analysis and cross-scale attention mechanisms.

## 📄 License

Academic research project.

