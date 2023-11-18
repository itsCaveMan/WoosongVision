# WoosongVision


## Setup 
Make sure python3 and pip and virtualenv are installed.

```

### Create virtual environment

# CD into the project directory
cd WoosongVision

# Create virtual environment
python3 virtualenv venv

# Activate virtual environment 
source venv/bin/activate

```


# Run example

```

# Run ai_system/woosongvision.py
python3 ai_system/woosongvision.py

```


# Training
all training related code is found in ai_system/training




# Dataset

The dataset is located in ai_system/dataset

Dataset was labeled using Roboflow

Dataset is publicly available at https://universe.roboflow.com/woosongvision/trashbags-segmentation/dataset/2

dataset_v1_Segmented_raw146(argmented593)_PytorchLabels:
- segmented labels
- 146 original images
- 593 total images after augmentation is applied


# Models
Models are located in ai_system/models

v2__YOLOv8s_Time2hGPU_train630val19_640x640:
- YOLOv8s - small
- Segmentation model
- roughly 2 hours of training on GPU




