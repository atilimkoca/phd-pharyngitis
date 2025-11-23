# Multi-Model Comparison Script

## Overview

`train_multi_model_comparison.py` is a comprehensive training script that tests multiple architectures with a for loop, comparing baseline models, hybrid attention models, and MSAN models with different backbones.

## Models Tested

### 1. Baseline Models (No Hybrid Attention)
- **ResNet50 Baseline** - Standard ResNet50 with binary classification head
- **AlexNet Baseline** - Standard AlexNet with binary classification head
- **MobileNetV2 Baseline** - Standard MobileNetV2 with binary classification head

### 2. Hybrid FGCR+CSBA Models (With Advanced Attention)
- **Hybrid FGCR+CSBA (ResNet50)** - ResNet50 + Frequency-Gated Channel Recalibration + Cross-Scale Bi-Attention
- **Hybrid FGCR+CSBA (AlexNet)** - AlexNet + FGCR + CSBA
- **Hybrid FGCR+CSBA (MobileNetV2)** - MobileNetV2 + FGCR + CSBA

### 3. MSAN Models (Multi-Scale Anatomical Attention Network)
- **MSAN (EfficientNet-B3)** - EfficientNet-B3 + Anatomical Region Proposal + Graph Attention + Frequency-Domain Inflammation Detection
- **MSAN (ResNet50)** - ResNet50 + MSAN components
- **MSAN (AlexNet)** - AlexNet + MSAN components
- **MSAN (MobileNetV2)** - MobileNetV2 + MSAN components

**Note:** MSAN models require PyWavelets. Install with: `pip install PyWavelets`

## Features

### Configurable Backbones
All hybrid and MSAN models support multiple backbones through a simple configuration parameter:

```python
# Hybrid models
model = HybridFGCR_CSBA(backbone='resnet50', pretrained=True)
model = HybridFGCR_CSBA(backbone='alexnet', pretrained=True)
model = HybridFGCR_CSBA(backbone='mobilenet_v2', pretrained=True)

# MSAN models
model = MSAN(backbone='efficientnet_b3', num_regions=5, pretrained=True)
model = MSAN(backbone='resnet50', num_regions=5, pretrained=True)
```

### Automatic Training Loop
The script automatically trains all models in sequence (up to 10 models if PyWavelets is installed):

```python
models_to_test = [
    ('ResNet50 Baseline', lambda: BaselineBinary('resnet50', pretrained=True)),
    ('AlexNet Baseline', lambda: BaselineBinary('alexnet', pretrained=True)),
    ('MobileNetV2 Baseline', lambda: BaselineBinary('mobilenet_v2', pretrained=True)),
    ('Hybrid FGCR+CSBA (ResNet50)', lambda: HybridFGCR_CSBA('resnet50', pretrained=True)),
    ('Hybrid FGCR+CSBA (AlexNet)', lambda: HybridFGCR_CSBA('alexnet', pretrained=True)),
    ('Hybrid FGCR+CSBA (MobileNetV2)', lambda: HybridFGCR_CSBA('mobilenet_v2', pretrained=True)),
]
```

## Usage

### Basic Usage

```bash
python train_multi_model_comparison.py
```

### Requirements

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- Pillow >= 9.3.0
- tqdm >= 4.64.0
- matplotlib >= 3.6.0
- openpyxl >= 3.0.0

### Dataset Structure

Ensure your data follows this structure:
```
phd/
├── excel.xlsx                          # Labels file
├── data_image_pharyngitis_nature/      # Image directory
│   ├── 1/
│   │   └── image.jpg
│   ├── 2/
│   │   └── image.jpg
│   └── ...
└── train_multi_model_comparison.py
```

**excel.xlsx format:**
- Column 0: Folder name (e.g., "1", "2", "3")
- Column 1: Binary label (0 = Non-Bacterial, 1 = Bacterial)

## Output

### 1. Saved Models
Each model is saved as a `.pth` file:
- `best_resnet50_baseline.pth`
- `best_alexnet_baseline.pth`
- `best_mobilenetv2_baseline.pth`
- `best_hybrid_fgcr+csba_resnet50.pth`
- `best_hybrid_fgcr+csba_alexnet.pth`
- `best_hybrid_fgcr+csba_mobilenetv2.pth`

### 2. Training History Plots
PNG files for each model showing:
- Training and validation loss
- Training and validation accuracy
- Validation AUC progression

Examples:
- `training_history_resnet50_baseline.png`
- `training_history_hybrid_fgcr+csba_resnet50.png`

### 3. Results Comparison Table
`model_comparison_results.xlsx` contains:
- Model name
- Total parameters
- Best validation AUC
- Test accuracy, precision, recall, F1-score
- Test AUC, specificity, sensitivity

### 4. Console Output
Final comparison table printed to console:

```
======================================================================
FINAL RESULTS COMPARISON
======================================================================

Model                           Parameters  Best Val AUC  Test Accuracy  Test AUC  ...
ResNet50 Baseline              25,557,032        0.8523         0.8421    0.8456
AlexNet Baseline               57,012,745        0.8201         0.8132    0.8089
MobileNetV2 Baseline            3,504,872        0.8389         0.8298    0.8312
Hybrid FGCR+CSBA (ResNet50)    28,142,664        0.9124         0.8978    0.9056
Hybrid FGCR+CSBA (AlexNet)     59,321,432        0.8812         0.8745    0.8823
Hybrid FGCR+CSBA (MobileNetV2)  5,892,544        0.8956         0.8834    0.8901
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BATCH_SIZE | 16 | Batch size for training |
| NUM_EPOCHS | 30 | Maximum epochs (early stopping may trigger) |
| PATIENCE | 7 | Early stopping patience |
| LEARNING_RATE | 0.0001 | Initial learning rate |
| IMG_SIZE | 224 | Input image size |

## Model Architecture Details

### Baseline Models
- Standard pretrained backbone
- Binary classification head (1 output neuron)
- BCE with Logits Loss
- No attention mechanisms

### Hybrid FGCR+CSBA Models

#### Components:

1. **Pretrained Backbone** (ResNet50/AlexNet/MobileNetV2)
   - Feature extraction layers
   - Adapters for non-ResNet backbones to match dimensions

2. **FGCR (Frequency-Gated Channel Recalibration)**
   - Applied at Layer 3 and Layer 4
   - FFT-based frequency analysis
   - Low/high frequency band separation
   - Channel recalibration weights
   - Spectral token generation (2-dim)

3. **CSBA (Cross-Scale Bi-Attention)**
   - Multi-head attention (4 heads, 128-dim)
   - Bi-directional attention between scales
   - Bridge token generation (128-dim)

4. **Classifier Head**
   - Input: Spatial features (2048) + Spectral token (2) + Bridge token (128)
   - Total: 2178 dimensions
   - MLP: 2178 → 512 → 1
   - Dropout: 0.5

## Training Strategy

### Loss Function
- **BCEWithLogitsLoss** with automatic class weighting
- `pos_weight = neg_count / pos_count` (handles class imbalance)

### Optimizer
- **Adam** optimizer
- Learning rate: 0.0001
- Weight decay: 1e-4

### Learning Rate Schedulers
1. **ReduceLROnPlateau**
   - Mode: maximize (AUC)
   - Factor: 0.5
   - Patience: 3 epochs

2. **CosineAnnealingWarmRestarts**
   - T_0: 10 epochs
   - T_mult: 2

### Early Stopping
- Monitors validation AUC
- Patience: 7 epochs
- Saves best model checkpoint

## Evaluation Metrics

For each model, the following metrics are calculated:

- **Accuracy** - Overall correctness
- **Precision** - Positive predictive value
- **Recall** (Sensitivity) - True positive rate
- **F1-Score** - Harmonic mean of precision and recall
- **AUC-ROC** - Area under the ROC curve
- **Specificity** - True negative rate

## Expected Performance

Based on similar pharyngitis classification tasks:

| Model Type | Expected Test AUC | Expected Test Accuracy |
|------------|-------------------|------------------------|
| ResNet50 Baseline | 0.82-0.86 | 0.80-0.85 |
| AlexNet Baseline | 0.78-0.82 | 0.76-0.81 |
| MobileNetV2 Baseline | 0.80-0.84 | 0.79-0.83 |
| **Hybrid (ResNet50)** | **0.88-0.92** | **0.86-0.90** |
| **Hybrid (AlexNet)** | **0.85-0.88** | **0.84-0.87** |
| **Hybrid (MobileNetV2)** | **0.86-0.90** | **0.85-0.88** |

## Customization

### Adding More Models

To add a new model to the comparison:

```python
# Define your model
class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return output

# Add to models_to_test list
models_to_test.append(
    ('My Custom Model', lambda: MyCustomModel().to(device))
)
```

### Modifying Hyperparameters

Edit the hyperparameters section at the top of the script:

```python
BATCH_SIZE = 32          # Larger batch (requires more GPU memory)
NUM_EPOCHS = 50          # More training epochs
PATIENCE = 10            # More patience for early stopping
LEARNING_RATE = 0.0005   # Higher learning rate
```

### Selecting Specific Models

Comment out models you don't want to test:

```python
models_to_test = [
    ('ResNet50 Baseline', lambda: BaselineBinary('resnet50', pretrained=True)),
    # ('AlexNet Baseline', lambda: BaselineBinary('alexnet', pretrained=True)),  # Commented out
    ('Hybrid FGCR+CSBA (ResNet50)', lambda: HybridFGCR_CSBA('resnet50', pretrained=True)),
]
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 8 or 4)
- Use gradient accumulation
- Test models one at a time

### Slow Training
- Increase `num_workers` in DataLoader (default: 4)
- Use mixed precision training
- Enable cudnn benchmarking

### Poor Performance
- Check class balance in dataset
- Verify image quality and labels
- Increase training epochs
- Adjust learning rate
- Try different data augmentation

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pharyngitis_classification_2025,
  title={Multi-Model Comparison Framework for Pharyngitis Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pharyngitis-classification}
}
```

## License

Academic research project.
