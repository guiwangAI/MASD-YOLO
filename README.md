# MASD-YOLO

<p align="center">
  <b>MASD-YOLO: A Ship Detection Algorithm with Cross-Modal Generalization in Remote Sensing Images</b>
</p>

<p align="center">
  <a href="https://github.com/guiwangAI/MASD-YOLO">
    <img src="https://img.shields.io/badge/Code-GitHub-blue.svg" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## Introduction

MASD-YOLO is a lightweight object detection framework for ship detection in remote sensing images. It is built on the YOLOv8 detection pipeline and aims to improve small-target representation, dense-ship localization, background-interference suppression, and cross-modal generalization between optical and SAR remote sensing images.

Remote sensing ship detection is challenging because ships often appear as small, densely distributed, and scale-varying targets. In addition, optical and SAR images have different imaging characteristics, including background noise, strong clutter, low contrast, and nearshore interference. MASD-YOLO addresses these issues by introducing three key modules:

- **ReLConv**: a reparameterized lightweight convolution module for efficient feature extraction and reduced computational cost.
- **EAFB**: an enhanced attention feature block that jointly models spatial and channel dependencies to suppress background interference.
- **MASDHead**: a multi-scale adaptive spatial decoupled detection head that improves geometric adaptation for ships with different scales and aspect ratios.

## Overall Architecture

The overall architecture of MASD-YOLO is shown below.

<p align="center">
  <img src="architecture.jpg" width="900">
</p>

<p align="center">
  <b>Figure 1.</b> Overall architecture of the proposed MASD-YOLO framework.
</p>

## Repository Structure

```text
MASD-YOLO/
├── MASD-YOLO.yaml          # Model configuration
├── ReLConv.py              # Reparameterized lightweight convolution module
├── EAFB.py                 # Enhanced attention feature block
├── MASDHead.py             # Multi-scale adaptive spatial decoupled head
├── train.py                # Training script
├── val.py                  # Validation script
├── predict.py              # Inference script
├── robustness.py           # Robustness dataset generation script
├── split_dataset.py        # Dataset splitting script
├── dataset.yaml            # Dataset configuration template
├── environment.yaml        # Conda environment file
├── architecture.jpg        # Overall architecture figure
├── attention/              # Attention modules used for comparison experiments
├── RSSVG/                  # RSSVG dataset
├── RSSVG-Rob/              # Robustness evaluation dataset
├── xView3/                 # xView3 dataset
└── ultralytics/            # Modified YOLOv8/Ultralytics framework
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/guiwangAI/MASD-YOLO.git
cd MASD-YOLO
```

### 2. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate MASD-YOLO
```

The provided environment uses Python 3.9, PyTorch 2.0.0, torchvision 0.15.1, and CUDA 11.8.

### 3. Check the installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"
```

## Dataset Preparation

The dataset should follow the YOLO detection format:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file should use the YOLO format:

```text
class_id x_center y_center width height
```

An example `dataset.yaml` is provided below:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: ship
```

Please update the `path` field according to your local dataset location.

## Training

Train MASD-YOLO with:

```bash
python train.py \
  --model ultralytics/cfg/models/v8/masd-yolo.yaml \
  --data dataset.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/train \
  --name masd-yolo
```

The trained weights will be saved to:

```text
runs/train/masd-yolo/weights/
```

## Validation

Evaluate a trained checkpoint with:

```bash
python val.py \
  --weights runs/train/masd-yolo/weights/best.pt \
  --data dataset.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/val \
  --name masd-yolo
```

## Inference

Run prediction on an image, folder, video, or stream:

```bash
python predict.py \
  --weights runs/train/masd-yolo/weights/best.pt \
  --source path/to/images_or_video \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  --project runs/predict \
  --name masd-yolo
```

The visualization results will be saved in:

```text
runs/predict/masd-yolo/
```

## Main Components

### ReLConv

`ReLConv` is designed to improve feature extraction efficiency by combining reparameterized convolution and lightweight downsampling. It replaces standard convolutional operations in the backbone to reduce computational cost while maintaining feature representation ability.

### EAFB

`EAFB` introduces enhanced spatial-channel feature modeling to highlight ship-related features and suppress interference from complex remote sensing backgrounds, such as sea surfaces, ports, islands, nearshore regions, and speckle noise.

### MASDHead

`MASDHead` is a multi-scale adaptive spatial decoupled detection head. It uses adaptive spatial modeling and decoupled prediction branches to better capture ships with different sizes, aspect ratios, and spatial distributions.

## Attention Mechanism Comparison

The `attention/` folder contains several representative attention mechanisms used for the **Comparison of attention mechanisms performance** experiment. These attention modules are inserted into the same position as the proposed attention-related module in MASD-YOLO, so that different attention mechanisms can be compared under the same YOLOv8-based detection framework.

```text
attention/
├── CASATT.py
├── MAN.py
├── MSA.py
├── SHSA.py
└── TA.py
```

These files are used only for comparative experiments. They are not the default components of MASD-YOLO. The default MASD-YOLO architecture uses the proposed ReLConv, EAFB, and MASDHead modules.

## Robustness Evaluation

The `robustness.py` script can generate corrupted datasets for robustness testing. Supported corruption types include Gaussian noise, salt-and-pepper noise, motion blur, JPEG artifacts, downscaling, brightness changes, low contrast, shadow, and cutout.

Example:

```bash
python robustness.py \
  --src-root RSSVG \
  --dst-root RSSVG-Rob \
  --seed 42 \
  --overwrite
```

To generate only selected corruption types:

```bash
python robustness.py \
  --src-root RSSVG \
  --dst-root RSSVG-Rob \
  --only gaussian_noise motion_blur low_contrast \
  --overwrite
```

## Dataset Splitting

If your dataset has not been split into training and validation sets, use:

```bash
python split_dataset.py \
  --src path/to/original_dataset \
  --out path/to/split_dataset \
  --val-ratio 0.2 \
  --seed 42 \
  --mode copy
```

## Deployment Environment on OrangePi AIpro

MASD-YOLO can be deployed on the OrangePi AIpro development board for edge-side ship detection. This section describes the environment preparation process. The training stage is recommended to be conducted on a desktop GPU server, while the OrangePi AIpro board is mainly used for edge-side inference and deployment verification.

The OrangePi AIpro development board follows the Ascend AI software stack. The official MindSpore tutorial provides instructions for image burning, CANN installation or upgrading, MindSpore installation or upgrading, and runtime environment configuration.

Official references:

- OrangePi AIpro environment setup: https://www.mindspore.cn/docs/en/r2.4.0/orange_pi/environment_setup.html
- OrangePi AIpro online inference tutorial: https://www.mindspore.cn/docs/en/r2.4.0/orange_pi/model_infer.html
- Orange Pi official website: https://www.orangepi.org/

### 1. Prepare the system image

Download the official OrangePi AIpro system image according to the board version. Burn the image to a Micro SD card using tools such as balenaEtcher or Rufus.

After burning the image, insert the Micro SD card into the OrangePi AIpro board, power on the board, and connect it to the network. SSH login can be used for remote development.

### 2. Check the CANN environment

After logging into the board, check whether the Ascend CANN toolkit has been installed:

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
```

Load the Ascend environment variables:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

It is recommended to add the environment configuration to `~/.bashrc`:

```bash
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

### 3. Check the NPU device

Use the following command to check whether the Ascend NPU is correctly recognized:

```bash
npu-smi info
```

If the NPU information can be displayed correctly, the hardware driver and runtime environment are available.

### 4. Check MindSpore

Check the installed MindSpore version:

```bash
pip show mindspore
```

Verify whether MindSpore can run on the Ascend backend:

```bash
python -c "import mindspore; mindspore.set_context(device_target='Ascend'); mindspore.run_check()"
```

If the check is successful, the MindSpore Ascend environment has been configured correctly.

### 5. Install basic Python dependencies

Install basic Python packages for image preprocessing and result visualization:

```bash
pip install numpy opencv-python pillow tqdm pyyaml
```

If OpenCV causes GUI-related dependency errors on the development board, use the headless version instead:

```bash
pip install opencv-python-headless
```

### 6. Prepare MASD-YOLO files

Clone this repository on the OrangePi AIpro board:

```bash
git clone https://github.com/guiwangAI/MASD-YOLO.git
cd MASD-YOLO
```

For quick functional verification, CPU inference can be tested first. For Ascend NPU inference, the trained model should be exported and converted according to the Ascend/MindSpore inference toolchain. Please make sure that the CANN version, MindSpore version, and model conversion toolchain are compatible.

### 7. Deployment notes

- The PyTorch training environment and the OrangePi AIpro inference environment are different.
- If NPU acceleration is required, the trained model needs to be converted to a format supported by the Ascend/MindSpore inference toolchain.
- If the model contains custom operators or unsupported layers, additional operator adaptation may be required during model conversion.
- The current repository provides the MASD-YOLO training, validation, inference, robustness evaluation, and environment preparation files. Hardware-specific model conversion scripts can be added according to the final deployment backend.

## Acknowledgements

This project is developed based on the Ultralytics YOLOv8 framework. We thank the open-source community for providing useful tools and resources for object detection research.

## License

This repository is released under the MIT License.
