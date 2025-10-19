Dsm4GdmSE: A Dual-stage Segmentation Model for Grape Downy Mildew Severity Evaluation





https://github.com/user-attachments/assets/e05af7fc-8845-4dc5-8f16-1f107f29ba6f

## Prerequisites

### Clone the Repository

Before starting with dataset preparation or model training, clone the repository by running:

### Environment Setup

The environment is developed with **Python 3.8**. To set up the environment, please install the required dependencies by running:

```
pip install -r requirements.txt
```
## Data
Click here to download the dataset
📂 Data

Click here
 to download the dataset.

The dataset consists of grape leaf images captured under natural field conditions, including both healthy and diseased samples.
Each image is manually annotated for leaf regions and downy mildew lesions, enabling dual-stage segmentation training.

##🧠 Model Overview

This study designs a dual-stage segmentation framework for grape downy mildew severity evaluation, composed of:

Leaf segmentation model – used to extract the complete grape leaf region.

Lesion segmentation model – used to segment disease lesions from the extracted leaf area.

###🍃 Leaf Segmentation Models

In the leaf segmentation stage, the following five models were trained and compared:

Model	Description
U-Net	Classic encoder–decoder architecture for semantic segmentation.
HRNet	High-resolution network maintaining spatial precision across stages.
PSPNet	Pyramid scene parsing network with global context aggregation.
DeepLabV3+	Encoder–decoder with atrous convolution and spatial pyramid pooling.
SegFormer	Transformer-based lightweight segmentation model.

These models aim to achieve precise extraction of grape leaf contours and reduce background interference for subsequent lesion segmentation.

###🍇 Lesion Segmentation Models

In the lesion segmentation stage, we built upon DeepLabV3+ as the baseline framework and explored multiple backbone networks to enhance feature extraction efficiency and lightweight design.

Backbone Variants
Backbone	Description
EfficientNetB0	Balanced network scaling with high efficiency.
MobileNetV4	Lightweight convolutional network optimized for mobile inference.
MobileNetV3	Combines inverted residuals with SE attention for better performance.
MobileNetV2	Efficient depthwise separable convolutions for compact representation.
StarNet	Custom backbone emphasizing multi-scale feature fusion.
Sim_MobileNetV3	Simplified MobileNetV3 variant optimized for fast convergence.
🔬 Additional Backbone Extensions

To further improve fine-grained lesion segmentation, additional experiments were conducted with classical CNN architectures:

Backbone	Notes
EfficientNetB0	Used as baseline for comparison.
VGG	Deep hierarchical structure emphasizing low-level detail extraction.
ResNet / ResNetRS50	Residual learning to stabilize deeper network training.
MobileNetV3	Lightweight and suitable for real-time inference.
MobileNetV2	Efficient and low-parameter baseline model.

These backbone variants were integrated with optimized convolution modules, focusing on feature sparsity, cross-layer fusion, and channel attention refinement to achieve higher lesion boundary precision and robust generalization in complex field environments.
