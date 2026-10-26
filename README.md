# SkinSeg-UNet: Deep Learning Skin Lesion Segmentation

A custom U-Net segmentation pipeline trained on the ISIC 2016 dataset to perform automated skin cancer lesion boundary extraction. The model incorporates smooth Swish activation functions, a hybrid BCE-Dice loss formulation, and an area-threshold post-processing filter to minimize false positives on healthy skin.

## Key Features
- **Architecture**: Custom U-Net with Average Pooling encoder blocks and skip-connection concatenation.
- **Activation**: Swish activation ($x \cdot \text{sigmoid}(\beta x)$) for smooth gradient flow along subtle lesion boundaries.
- **Loss Function**: Combined Binary Cross-Entropy (BCE) + Dice Loss for balanced pixel-level and region-level optimization.
- **Post-Processing Filter**: 10% area-thresholding rule that reduced healthy skin false-positive rates from 100% to 25%.
- **Performance**: Achieved **75.8% Test Dice Score** and **90.26% Pixel Accuracy** on unseen test data.

## Pipeline Breakdown

### 1. Data Pre-Processing
- All images and ground-truth masks are resized to $256 \times 256$ pixels.
- Pixel normalization scales raw values from [0-255] to [0.0-1.0].
- Masks are binarized using a threshold of $>127$.
- Dataset split: **70% Training** (630 pairs), **15% Validation** (135 pairs), and **15% Testing** (135 pairs).

### 2. Model Architecture & Training
- **Encoder**: 2D Convolutions combined with Average Pooling.
- **Decoder**: UpSampling2D to restore feature maps to $256 \times 256$ resolution.
- **Optimization**: Trained for 10 epochs with a batch size of 8.
- Validation Dice score improved from **56.63%** (Epoch 1) to **75.89%** (Epoch 10).

### 3. Evaluation Metrics
- **Test Dice Score**: 75.8%
- **Pixel Accuracy**: 90.26%
- **Confusion Matrix (8.8M Test Pixels)**:
  - True Negatives (Background Skin): $6,082,186$
  - True Positives (Lesion Boundary): $1,903,583$

### 4. Healthy Skin Noise Reduction
- **Problem**: Model misidentifies skin folds or light reflections as tiny lesions on healthy skin.
- **Solution**: Implemented a 10% area-thresholding post-processing filter. If predicted lesion coverage is $<10\%$, the mask output is zeroed out as background noise.
- **Result**: Lowered healthy skin False Positive Rate from 100% to 25%.
