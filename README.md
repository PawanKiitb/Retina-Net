# Retina-Net

**An Object Detection Model Trained on the Stanford Drone Dataset**

![Python](https://img.shields.io/badge/Made%20with-Python-blue)

## Overview

This repository implements **Retina-Net**, an object detection model, trained on the **Stanford Drone Dataset**. The model is customized to detect aerial pedestrians in drone-captured imagery, achieving a mean Average Precision (mAP) of **0.49**.

Key features:
- **Backbone**: ResNet50
- **Neck**: Feature Pyramid Network (FPN)
- **Loss Function**: Focal Loss
- **Custom Anchors**: Adjusted for small pedestrian sizes

---

## Features

- **Dataset**: Utilizes the Stanford Drone Dataset.
- **Custom Anchors**: Optimized for small pedestrian bounding boxes.
- **Transforms**: Data augmentation with Albumentations library.
- **Visualization**: Tools for visualizing augmentations and model predictions.
- **Training & Inference Utilities**: Includes scripts for training, validation, and inference.

---

## Table of Contents

- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Results](#results)
- [Code Structure](#code-structure)

---

## Requirements

- Python 3.x
- Libraries:
  - PyTorch
  - Albumentations
  - OpenCV
  - TorchMetrics
  - tqdm
- GPU Recommended

---

## Dataset Setup

1. Download the **Stanford Drone Dataset** from [http://cvgl.stanford.edu/projects/uav_data/](http://cvgl.stanford.edu/projects/uav_data/).
2. Place the dataset in the `data/` directory:
   ```
   data/
   ├── imgs/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── annotations/
   │   ├── train_annotations.csv
   │   ├── val_annotations.csv
   │   └── test_annotations.csv
   └── labels.csv
   ```

---

## Usage

### Installation

Clone the repository:

```bash
git clone https://github.com/PawanKiitb/Retina-Net.git
cd Retina-Net
```

### Training

To train the Retina-Net model:

```bash
python3 main.py
```

Adjust configurations in `config.py` as needed:
- `NUM_EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate
- `DEVICE`: Specify `cuda` for GPU or `cpu`

### Evaluation

Evaluate the model on the validation set to compute the **mAP**:

```bash
python3 src/utils/trainer.py
```

### Inference

Run inference on test images:

```bash
python3 src/utils/inference.py
```
Results will be stored in metrics.csv for each epoch, with its corresponding mAP and loss.
---

## Results

The model achieved the following results on the **Stanford Drone Dataset**:

- **Mean Average Precision (mAP)**: 0.49

---

## Other Utilitis

visualizeTransforms.py can be used to visualize the current transforms applied on the image during preprocessing(It draws from the src/Dataset/transformsVisualization.py).

Outputs can be visualized with bounding boxes using the visualizeResult.py in utils in src.

One can change the model or anchor-size generator design or Classifier Head implementation, according to their needs from models in src module

## Code Structure

```
Retina-Net/
├── config.py                 # Configuration file for training
├── main.py                   # Main training entry point
├── metrics.csv               # Training metrics log
├── data/                     # Dataset directory
├── src/
│   ├── Dataset/
│   │   ├── dataset.py        # Dataset loader
│   │   ├── transforms.py     # Data augmentation transforms
│   │   ├── transformsVisualization.py # Augmentation visualization
│   ├── models/
│   │   ├── retinaNet.py      # Retina-Net model definition
│   ├── utils/
│   │   ├── collateFunction.py # Batch collation for DataLoader
│   │   ├── trainer.py        # Training and validation loops
│   │   ├── inference.py      # Inference script
│   │   ├── aspectRatio.py    # Bounding box aspect ratio calculations
├── visualizeTransforms.py    # Visualization of augmentations
└── requirements.txt          # Python dependencies
```

---
