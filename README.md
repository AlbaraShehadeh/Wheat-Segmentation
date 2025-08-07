# 🌾 Wheat Segmentation Project

A comprehensive computer vision project implementing state-of-the-art segmentation models for wheat head detection and segmentation in field images. This project explores both **SAM (Segment Anything Model)** fine-tuning and **YOLOv8** segmentation approaches to achieve high-precision wheat head identification.

![Project Banner](https://img.shields.io/badge/Computer%20Vision-Wheat%20Segmentation-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)

## 📋 Table of Contents

- [🌾 Wheat Segmentation Project](#-wheat-segmentation-project)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Project Overview](#-project-overview)
  - [🔧 Features](#-features)
  - [📂 Dataset Structure](#-dataset-structure)
  - [🏗️ Project Structure](#️-project-structure)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [🧠 Models \& Approaches](#-models--approaches)
    - [1. SAM (Segment Anything Model) Fine-tuning](#1-sam-segment-anything-model-fine-tuning)
    - [2. YOLOv8 Segmentation Models](#2-yolov8-segmentation-models)
  - [📊 Model Performance](#-model-performance)
  - [💻 Usage](#-usage)
    - [Training SAM Model](#training-sam-model)
    - [Training YOLOv8 Models](#training-yolov8-models)
    - [Running Inference](#running-inference)
  - [📁 Pre-trained Models](#-pre-trained-models)
  - [🔄 Model Export \& Deployment](#-model-export--deployment)
  - [📈 Results](#-results)
  - [🛠️ Technical Details](#️-technical-details)
  - [📚 References](#-references)
  - [👥 Contributing](#-contributing)
  - [📄 License](#-license)

## 🎯 Project Overview

This project addresses the challenge of automated wheat head detection and segmentation in agricultural field images. Accurate wheat head counting and segmentation is crucial for:

- **Yield Estimation**: Predicting crop yield based on wheat head density
- **Agricultural Research**: Analyzing wheat variety performance
- **Precision Agriculture**: Optimizing farming practices through computer vision
- **Automated Monitoring**: Real-time crop assessment using drones or field cameras

## 🔧 Features

- ✅ **Dual Model Approach**: Implementation of both SAM and YOLOv8 for comprehensive comparison
- ✅ **High-Quality Dataset**: 1000 annotated wheat field images with precise segmentation masks
- ✅ **Multiple Model Variants**: YOLOv8m and YOLOv8l implementations for different performance requirements
- ✅ **ONNX Export**: Optimized models for production deployment
- ✅ **Comprehensive Training**: Complete training pipelines with visualization and metrics
- ✅ **Real-time Inference**: Fast inference capabilities for practical applications

## 📂 Dataset Structure

```
data/
├── train/
│   ├── images/          # 1000 wheat field images (.jpg)
│   └── masks/           # 1000 corresponding segmentation masks (.png)
└── test/                # 10 test images for evaluation
    ├── 2fd875eaa.jpg
    ├── 348a992bb.jpg
    └── ... (8 more test images)
```

**Dataset Specifications:**
- **Training Images**: 1000 high-resolution wheat field photographs
- **Image Format**: JPG (RGB)
- **Mask Format**: PNG (Binary masks)
- **Image Resolution**: Resized to 256×256 for SAM, 640×640 for YOLO
- **Annotation Quality**: Pixel-perfect segmentation masks for wheat heads

## 🏗️ Project Structure

```
Wheat-Segmentation-1/
├── data/                          # Dataset directory
├── SAM/                           # SAM model implementation
│   ├── fine-tune-sam.ipynb      # SAM fine-tuning notebook
│   └── model.pth                 # Trained SAM model weights
├── YOLO/                          # YOLO model implementations
│   ├── yolo8m/                   # YOLOv8 Medium variant
│   │   ├── train-yolov8m-seg.ipynb
│   │   ├── wheat_segmentation_model.onnx
│   │   └── runs/                 # Training results and metrics
│   └── yolov8l/                  # YOLOv8 Large variant
│       ├── train-yolov8l-seg.ipynb
│       └── runs/                 # Training results and metrics
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Wheat-Segmentation-1.git
cd Wheat-Segmentation-1
```

2. **Install dependencies**
```bash
# Install PyTorch (visit pytorch.org for specific CUDA version)
pip install torch torchvision torchaudio

# Install Ultralytics for YOLOv8
pip install ultralytics==8.0.196

# Install SAM dependencies
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/huggingface/transformers.git

# Additional dependencies
pip install datasets opencv-python matplotlib monai roboflow
pip install onnx onnxruntime  # For model export
```

3. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import ultralytics; ultralytics.checks()"
```

## 🧠 Models & Approaches

### 1. SAM (Segment Anything Model) Fine-tuning

**Approach**: Fine-tune Meta's Segment Anything Model on wheat segmentation data using bounding box prompts.

**Key Features:**
- **Base Model**: `facebook/sam-vit-large`
- **Training Strategy**: Freeze vision encoder and prompt encoder, fine-tune mask decoder only
- **Loss Function**: Focal Loss for handling class imbalance
- **Input Resolution**: 1024×1024 (SAM native resolution)
- **Prompt Strategy**: Bounding box prompts with perturbation for robustness

**Training Configuration:**
- **Epochs**: 5
- **Batch Size**: 4
- **Optimizer**: Adam (lr=1e-5)
- **Device**: Multi-GPU training with DataParallel

### 2. YOLOv8 Segmentation Models

**Approach**: Train YOLOv8 models specifically for wheat head instance segmentation.

#### YOLOv8m (Medium)
- **Parameters**: ~27M
- **Speed**: Balanced performance
- **Use Case**: General-purpose applications

#### YOLOv8l (Large)
- **Parameters**: ~46M
- **Speed**: Higher accuracy, slower inference
- **Use Case**: High-precision applications

**Training Configuration:**
- **Epochs**: 25 (YOLOv8m), 15 (YOLOv8l)
- **Image Size**: 640×640
- **Batch Size**: 16
- **Augmentations**: Mosaic, mixup, rotation, scaling

## 📊 Model Performance

### YOLOv8 Results

| Model | mAP50 (Box) | mAP50-95 (Box) | mAP50 (Mask) | mAP50-95 (Mask) | Speed (ms) |
|-------|-------------|----------------|--------------|-----------------|------------|
| YOLOv8m | 96.9% | 84.6% | 96.4% | 71.4% | ~1400 |
| YOLOv8l | 96.9% | 84.3% | 96.5% | 70.8% | ~76 |

### SAM Results
- **Training Loss Reduction**: From 0.45 to 0.16 over 5 epochs
- **Convergence**: Stable training with consistent improvement
- **Segmentation Quality**: High-quality masks with precise boundaries

## 💻 Usage

### Training SAM Model

```bash
# Navigate to SAM directory
cd SAM/

# Open and run the Jupyter notebook
jupyter notebook fine-tune-sam.ipynb
```

The SAM training process includes:
1. Data loading and preprocessing
2. Bounding box prompt generation
3. Model fine-tuning with Focal Loss
4. Visualization of results

### Training YOLOv8 Models

```bash
# For YOLOv8m
cd YOLO/yolo8m/
yolo task=segment mode=train model=yolov8m-seg.pt data=path/to/data.yaml epochs=25 imgsz=640

# For YOLOv8l
cd YOLO/yolov8l/
yolo task=segment mode=train model=yolov8l-seg.pt data=path/to/data.yaml epochs=15 imgsz=640
```

### Running Inference

#### YOLO Inference
```bash
# Using PyTorch model
yolo task=segment mode=predict model=path/to/best.pt source=path/to/images conf=0.25

# Using ONNX model
yolo task=segment mode=predict model=path/to/best.onnx source=path/to/images conf=0.25
```

#### SAM Inference
```python
from transformers import SamModel, SamProcessor
import torch

# Load the fine-tuned model
model = SamModel.from_pretrained("facebook/sam-vit-large")
model.load_state_dict(torch.load("SAM/model.pth"))
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")

# Run inference with bounding box prompt
inputs = processor(image, input_boxes=[[bbox]], return_tensors="pt")
outputs = model(**inputs)
```

## 📁 Pre-trained Models

### Available Models

| Model | Type | Size | Location | Performance |
|-------|------|------|----------|-------------|
| SAM Fine-tuned | PyTorch | ~2.4GB | `SAM/model.pth` | High-quality masks |
| YOLOv8m | PyTorch | ~52MB | `YOLO/yolo8m/runs/segment/train/weights/best.pt` | mAP50: 96.9% |
| YOLOv8l | PyTorch | ~88MB | `YOLO/yolov8l/runs/segment/train3/weights/best.pt` | mAP50: 96.9% |
| YOLOv8m | ONNX | ~105MB | `YOLO/yolo8m/wheat_segmentation_model.onnx` | Production-ready |
| YOLOv8l | ONNX | ~175MB | `YOLO/yolov8l/runs/segment/train3/weights/best.onnx` | Production-ready |

### Model Selection Guide

- **For Research/Experimentation**: Use SAM fine-tuned model for highest quality masks
- **For Real-time Applications**: Use YOLOv8m ONNX model for balanced speed/accuracy
- **For Maximum Accuracy**: Use YOLOv8l PyTorch model
- **For Production Deployment**: Use ONNX versions for cross-platform compatibility

## 🔄 Model Export & Deployment

### ONNX Export

```bash
# Export YOLOv8 to ONNX
from ultralytics import YOLO
model = YOLO("path/to/best.pt")
model.export(format="onnx", dynamic=False)
```

### Deployment Options

1. **Local Inference**: Direct PyTorch/ONNX inference
2. **Roboflow Deploy**: Cloud-based API deployment
3. **Edge Deployment**: ONNX Runtime for edge devices
4. **Docker Containers**: Containerized inference services

## 📈 Results

### Qualitative Results
- **Precision**: Models accurately identify individual wheat heads
- **Boundary Quality**: Sharp, precise segmentation boundaries
- **Robustness**: Consistent performance across different lighting conditions
- **Scalability**: Efficient processing of multiple wheat heads per image

### Quantitative Metrics
- **Detection Accuracy**: >96% mAP50 for both YOLO variants
- **Segmentation Quality**: >96% mask mAP50
- **Speed**: Real-time inference capability with optimized models
- **Generalization**: Strong performance on test set

## 🛠️ Technical Details

### Data Preprocessing
- **Image Normalization**: 0-1 scaling for neural networks
- **Augmentation**: Geometric and photometric augmentations
- **Mask Processing**: Binary mask conversion and validation
- **Quality Control**: Filtering of empty masks and corrupted data

### Training Optimizations
- **Mixed Precision**: Automatic Mixed Precision for faster training
- **Multi-GPU**: DataParallel training support
- **Learning Rate**: Cosine annealing and warmup strategies
- **Early Stopping**: Patience-based training termination

### Inference Optimizations
- **Model Fusion**: Layer fusion for faster inference
- **Batch Processing**: Efficient batch inference
- **Memory Management**: Optimized memory usage
- **Post-processing**: Efficient NMS and mask generation

## 📚 References

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Roboflow for Dataset Management](https://roboflow.com/)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🌾 Happy Wheat Segmentation!** 

For questions or support, please open an issue or contact the maintainers.