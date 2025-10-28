# Dsm4GdmSE: A Dual-stage Segmentation Model for Grape Downy Mildew Severity Evaluation





https://github.com/user-attachments/assets/e05af7fc-8845-4dc5-8f16-1f107f29ba6f

## Prerequisites

### Clone the Repository

Before starting with dataset preparation or model training, clone the repository by running:
```
git clone https://github.com/husnxjsjsb/Grape-Downy.git
```
Enter the project directory
```
cd grape-downy
```
### Environment Setup

The environment is developed with **Python 3.8**
```
conda create -n grape python=3.8
```

Activate the environment:
```
conda activate grape
```
To set up the environment, please install the required dependencies by running:
```
pip install -r requirements.txt
```
## Data
Click here to download the dataset
📂 [Data](https://pan.baidu.com/s/1PZzxG0q9FyyoeLAhU9tLVg?pwd=pb3d)



The dataset consists of grape leaf images captured under natural field conditions, including both healthy and diseased samples.
Each image is manually annotated for leaf regions and downy mildew lesions, enabling dual-stage segmentation training.

## 🧠 Model Overview

This study designs a dual-stage segmentation framework for grape downy mildew severity evaluation, composed of:

Leaf segmentation model – used to extract the complete grape leaf region.

Lesion segmentation model – used to segment disease lesions from the extracted leaf area.
Click here to download the [model](https://pan.baidu.com/s/1VBRBHVFbY_FSG2mHYHJJFQ?pwd=x9hw) 


## 🍃 Leaf Segmentation Models

In the leaf segmentation stage, multiple deep learning architectures were evaluated to ensure accurate leaf boundary extraction and minimal background interference.


| Model | Description |
|--------|--------------|
| **U-Net** | Classic encoder–decoder segmentation network. |
| **HRNet** | Maintains high-resolution feature maps across all stages. |
| **PSPNet** | Utilizes pyramid pooling for global context understanding. |
| **DeepLabV3+** | Combines atrous convolution with encoder–decoder design. |
| **SegFormer** | Lightweight transformer-based segmentation model. |

### 🧩 Backbone Variants
| Backbone | Description |
|-----------|--------------|
| **EfficientNetB0** | Balanced accuracy and efficiency through compound scaling. |
| **MobileNetV4** | Optimized lightweight model for mobile and edge inference. |
| **MobileNetV3** | Incorporates SE attention and inverted residuals for better feature reuse. |
| **MobileNetV2** | Efficient representation via depthwise separable convolutions. |
| **StarNet** | Custom multi-scale fusion backbone for agricultural imagery. |
| **Sim_MobileNetV3** | Simplified variant for faster convergence and smaller size. |

These models aim to achieve precise segmentation of grape leaf contours and robust generalization under complex lighting and background conditions.
If you want to replace the backbone network, you need to change this part of the code to the backbone you want:
```
backbone = "sim_mobilenetv3"
```
### 🍇 Lesion Segmentation Models

In the lesion segmentation stage, we built upon UNet as the baseline framework and designed an improved backbone network to enhance feature representation and model generalization. Specifically, EfficientNet was selected as the encoder to achieve a better balance between accuracy and computational efficiency.



To further improve fine-grained lesion segmentation, additional experiments were conducted with classical CNN architectures:

### 🍇 Backbone	Notes
| Backbone | Description |
|-----------|--------------|
| **EfficientNetB0** | Used as baseline for comparison. |
| **VGG** | Deep hierarchical structure emphasizing low-level detail extraction. |
| **ResNet / ResNetRS50** | Residual learning to stabilize deeper network training. |
| **MobileNetV3** | Lightweight and suitable for real-time inference. |
| **MobileNetV2** | Efficient and low-parameter baseline model. |


These backbone variants were integrated with optimized convolution modules, focusing on feature sparsity, cross-layer fusion, and channel attention refinement to achieve higher lesion boundary precision and robust generalization in complex field environments.
If you want to test a specific model, after downloading it, place the model file in the Model folder. Then, use the corresponding model’s get_miou code. Modify the :
```
"backbone": ""
```
Next, if you want to test the model results, open the efficientnet_pytorch package, then open EfficientNet, and replace the original backbone with src\leison\UNet\nets\unet\attention\sim.py. After that, run the following command:
```
python src\leison\UNet\main.py
```
field as needed based on your configuration, and you will be able to obtain the results.

## 💡How to Use the App

App Download Link：
