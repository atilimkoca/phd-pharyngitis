"""
Generate Visual Schematic for Hybrid Attention Model Architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_resnet = '#B8E6F5'
color_fgcr = '#FFE5CC'
color_csba = '#FFD1DC'
color_classifier = '#D5E8D4'
color_output = '#F4CCCC'

# Title
ax.text(5, 29, 'Hybrid Attention Model Architecture', 
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 28.3, 'ResNet50 + FGCR + CSBA for Binary Pharyngitis Classification', 
        ha='center', va='top', fontsize=12, style='italic')

# ==================== INPUT ====================
y_pos = 27
input_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.8, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor=color_input, linewidth=2)
ax.add_patch(input_box)
ax.text(5, y_pos-0.1, 'Input Image', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, y_pos-0.35, '224×224×3', ha='center', va='center', fontsize=9)

# Arrow down
ax.arrow(5, y_pos-0.6, 0, -0.4, head_width=0.2, head_length=0.1, fc='black', ec='black')

# ==================== RESNET BACKBONE ====================
y_pos = 25.5
# Conv1 + Early layers
conv1_box = FancyBboxPatch((3, y_pos-0.5), 4, 0.7, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_resnet, linewidth=1.5)
ax.add_patch(conv1_box)
ax.text(5, y_pos-0.15, 'Conv1 + BN + ReLU + MaxPool', ha='center', va='center', fontsize=9)

ax.arrow(5, y_pos-0.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# Layer 1
y_pos = 24.5
layer1_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_resnet, linewidth=1.5)
ax.add_patch(layer1_box)
ax.text(5, y_pos-0.1, 'Layer 1', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.35, '256 channels, 56×56', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-0.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# Layer 2
y_pos = 23.5
layer2_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_resnet, linewidth=1.5)
ax.add_patch(layer2_box)
ax.text(5, y_pos-0.1, 'Layer 2', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.35, '512 channels, 28×28', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-0.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# ==================== LAYER 3 + FGCR ====================
y_pos = 22.5
layer3_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_resnet, linewidth=1.5)
ax.add_patch(layer3_box)
ax.text(5, y_pos-0.1, 'Layer 3', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.35, '1024 channels, 14×14', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-0.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# FGCR-3
y_pos = 21.5
fgcr3_box = FancyBboxPatch((3, y_pos-0.8), 4, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='#FF8C00', facecolor=color_fgcr, linewidth=2)
ax.add_patch(fgcr3_box)
ax.text(5, y_pos, 'FGCR-3: Frequency Gating', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.25, '↓ FFT → Low/High Freq Split', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.45, '↓ Energy Calculation', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.65, '↓ Channel Recalibration', ha='center', va='center', fontsize=8)

# Save low scale (arrow to right)
ax.arrow(5, y_pos-0.4, 1.8, 0, head_width=0.12, head_length=0.15, fc='blue', ec='blue', linewidth=2)
ax.text(7.5, y_pos-0.3, 'LOW SCALE', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='blue', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(7.5, y_pos-0.6, '1024×14×14', ha='center', va='center', fontsize=7, color='blue')

ax.arrow(5, y_pos-0.9, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# ==================== LAYER 4 + FGCR ====================
y_pos = 19.8
layer4_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.6, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color_resnet, linewidth=1.5)
ax.add_patch(layer4_box)
ax.text(5, y_pos-0.1, 'Layer 4', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.35, '2048 channels, 7×7', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-0.6, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# FGCR-4
y_pos = 18.8
fgcr4_box = FancyBboxPatch((3, y_pos-0.8), 4, 1.2, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='#FF8C00', facecolor=color_fgcr, linewidth=2)
ax.add_patch(fgcr4_box)
ax.text(5, y_pos, 'FGCR-4: Frequency Gating', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.25, '↓ FFT → Low/High Freq Split', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.45, '↓ Energy Calculation', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.65, '↓ Channel Recalibration', ha='center', va='center', fontsize=8)

# Spectral token output (arrow to left)
ax.arrow(5, y_pos-0.4, -1.8, 0, head_width=0.12, head_length=0.15, fc='red', ec='red', linewidth=2)
ax.text(2.5, y_pos-0.3, 'Spectral Token', ha='center', va='center', fontsize=8, 
        fontweight='bold', color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(2.5, y_pos-0.55, '[E_low, E_high]', ha='center', va='center', fontsize=7, color='red')
ax.text(2.5, y_pos-0.75, '2-dim', ha='center', va='center', fontsize=7, color='red')

# HIGH SCALE label
ax.text(7.5, y_pos-0.3, 'HIGH SCALE', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(7.5, y_pos-0.6, '2048×7×7', ha='center', va='center', fontsize=7, color='green')

# ==================== CSBA ====================
y_pos = 17.2
# Arrow from low scale
ax.annotate('', xy=(5, y_pos+0.5), xytext=(7.5, y_pos+1.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

# Arrow continuing from high scale
ax.arrow(5, y_pos+0.6, 0, -0.25, head_width=0.15, head_length=0.08, fc='black', ec='black')

csba_box = FancyBboxPatch((2.5, y_pos-1.2), 5, 1.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#FF1493', facecolor=color_csba, linewidth=2.5)
ax.add_patch(csba_box)
ax.text(5, y_pos+0.2, 'Cross-Scale Bi-Attention (CSBA)', ha='center', va='center', 
        fontsize=11, fontweight='bold')

# CSBA details
ax.text(5, y_pos-0.15, '1. Flatten to tokens (low: 196, high: 49)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.35, '2. Embed to attention dimension (128-dim)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.55, '3. Multi-Head Attention (4 heads, bi-directional)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.75, '4. Project back & Add residual', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.95, '5. Generate Bridge Token (128-dim)', ha='center', va='center', fontsize=8)

# Bridge token output
ax.arrow(5, y_pos-1.3, -1.8, -0.5, head_width=0.12, head_length=0.15, fc='purple', ec='purple', linewidth=2)
ax.text(2.5, y_pos-2.1, 'Bridge Token', ha='center', va='center', fontsize=8, 
        fontweight='bold', color='purple', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(2.5, y_pos-2.35, '128-dim', ha='center', va='center', fontsize=7, color='purple')

# Enhanced features output
ax.arrow(5, y_pos-1.3, 0, -0.5, head_width=0.15, head_length=0.08, fc='black', ec='black')
ax.text(6.5, y_pos-1.6, 'Enhanced\nFeatures', ha='center', va='center', fontsize=8, fontweight='bold')

# ==================== GLOBAL AVERAGE POOLING ====================
y_pos = 14
gap_box = FancyBboxPatch((3.5, y_pos-0.4), 3, 0.6, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color_classifier, linewidth=1.5)
ax.add_patch(gap_box)
ax.text(5, y_pos-0.05, 'Global Average Pooling', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.28, '2048×7×7 → 2048', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-0.5, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# ==================== FEATURE CONCATENATION ====================
y_pos = 13
concat_box = FancyBboxPatch((2, y_pos-0.6), 6, 0.9, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#FFF9E6', linewidth=2)
ax.add_patch(concat_box)
ax.text(5, y_pos-0.05, 'Feature Concatenation', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5, y_pos-0.35, '[Spatial (2048) + Spectral (2) + Bridge (128)]', ha='center', va='center', fontsize=9)
ax.text(5, y_pos-0.52, 'Total: 2178 dimensions', ha='center', va='center', fontsize=8, style='italic')

# Arrows from tokens
ax.annotate('', xy=(4, y_pos-0.2), xytext=(2.5, y_pos-1.3),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
ax.annotate('', xy=(3.5, y_pos-0.25), xytext=(2.5, y_pos-3.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))

ax.arrow(5, y_pos-0.7, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# ==================== CLASSIFIER ====================
y_pos = 11.5
classifier_box = FancyBboxPatch((3, y_pos-1.2), 4, 1.8, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color_classifier, linewidth=2)
ax.add_patch(classifier_box)
ax.text(5, y_pos+0.2, 'Classifier Head (MLP)', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, y_pos-0.1, '↓ Dropout (0.5)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.3, '↓ Linear (2178 → 512)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.5, '↓ ReLU', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.7, '↓ Dropout (0.5)', ha='center', va='center', fontsize=8)
ax.text(5, y_pos-0.9, '↓ Linear (512 → 1)', ha='center', va='center', fontsize=8)

ax.arrow(5, y_pos-1.3, 0, -0.3, head_width=0.15, head_length=0.08, fc='black', ec='black')

# ==================== OUTPUT ====================
y_pos = 9.3
output_box = FancyBboxPatch((3.5, y_pos-0.5), 3, 0.8, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color_output, linewidth=2)
ax.add_patch(output_box)
ax.text(5, y_pos-0.05, 'Binary Output', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, y_pos-0.3, 'Sigmoid → [0, 1]', ha='center', va='center', fontsize=9)

# Final predictions
y_pos = 8.2
ax.text(3.5, y_pos, '0 = Non-Bacterial', ha='left', va='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(6.5, y_pos, '1 = Bacterial', ha='right', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ==================== LEGEND ====================
y_pos = 6.5
ax.text(5, y_pos+0.3, 'Component Legend', ha='center', va='center', fontsize=12, fontweight='bold')

legend_items = [
    (color_resnet, 'ResNet Backbone'),
    (color_fgcr, 'FGCR Module'),
    (color_csba, 'CSBA Module'),
    (color_classifier, 'Pooling/Classifier'),
]

for i, (color, label) in enumerate(legend_items):
    x_pos = 2.5 + (i % 2) * 3
    y_offset = y_pos - 0.3 - (i // 2) * 0.5
    legend_box = FancyBboxPatch((x_pos-0.5, y_offset-0.15), 1.8, 0.3, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor=color, linewidth=1)
    ax.add_patch(legend_box)
    ax.text(x_pos+0.4, y_offset, label, ha='center', va='center', fontsize=8)

# ==================== KEY INNOVATIONS ====================
y_pos = 4
ax.text(5, y_pos+0.3, 'Key Innovations', ha='center', va='center', fontsize=12, fontweight='bold')

innovations = [
    '✓ Frequency-Domain Channel Recalibration',
    '✓ Bi-Directional Cross-Scale Attention',
    '✓ Multi-Source Features (Spatial + Spectral + Cross-Scale)',
]

for i, innovation in enumerate(innovations):
    ax.text(5, y_pos-0.2-i*0.35, innovation, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ==================== STATISTICS ====================
y_pos = 2
ax.text(5, y_pos+0.3, 'Model Statistics', ha='center', va='center', fontsize=12, fontweight='bold')

stats = [
    'Parameters: ~28M  |  FLOPs: ~8.2G  |  Input: 224×224×3  |  Output: Binary',
]

ax.text(5, y_pos-0.1, stats[0], ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ==================== FOOTER ====================
y_pos = 0.8
ax.text(5, y_pos, 'Hybrid Attention Model: Combining ResNet50 + FGCR + CSBA', 
        ha='center', va='center', fontsize=10, style='italic')
ax.text(5, y_pos-0.4, 'For Binary Pharyngitis Classification (Bacterial vs Non-Bacterial)', 
        ha='center', va='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('model_architecture_schematic.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Schematic saved as 'model_architecture_schematic.png'")
plt.close()
