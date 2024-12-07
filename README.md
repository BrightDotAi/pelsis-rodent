# Rodent Detection with Faster R-CNN

  

This repository contains the code for training and evaluating a rodent detection model using Faster R-CNN with a ResNet-50 backbone. The model is trained on custom rodent detection data and used for inference on new images. The project leverages PyTorch and Torchvision to implement the Faster R-CNN model.

  

## Table of Contents

  

- [Rodent Detection with Faster R-CNN](#rodent-detection-with-faster-r-cnn)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Inference](#inference)
    - [Dataset](#dataset)
      - [Model Details](#model-details)
      - [Model Training](#model-training)
      - [Inference](#inference-1)
  - [Notes](#notes)
  - [Licenses](#licenses)

  

## Installation

  

 To get started, clone this repository and set up the necessary dependencies.



  

 1. Clone the repository:


    git clone https://github.com/your-username/rodent-detection.git
    
    cd rodent-detection
 2. Create a new Python virtual environment (optional but recommended):

  
    python3 -m venv venv
    
    source venv/bin/activate # On Windows, use venv\Scripts\activate

 3. Install the required dependencies:

    pip install -r requirements.txt

The requirements.txt file should include essential libraries like PyTorch, Torchvision, OpenCV, and other dependencies.

  

## Usage

### Training

To train the rodent detection model, run the build.py script. This will load your training data, train the model, and save the trained weights to a specified directory.
  

    python build.py

Arguments

The script assumes the presence of a custom annotations file in COCO format (e.g., instances_default.json).

The training data should be stored in the data/images directory.

The model will be saved in the models/trained_model directory after each epoch.

The script will output loss and other metrics during training. You can specify the batch size and other parameters directly in the script if needed.

  

### Inference

To run inference and visualize the results, use the eval.py script. This script performs object detection on new images and saves the output images with bounding boxes. It also generates a CSV file with image names and the corresponding true and predicted labels.


    python eval.py

Arguments

The script assumes that the trained model (model_epoch_X.pth) is available in the models/trained_model directory.

The annotations file is loaded from data/instances_default.json (for inference, this is used to match file names).

The script will output:

Images with drawn bounding boxes saved in the outputs/ folder.

A CSV file (predictions.csv) containing the image name, true label, and predicted label.

Example CSV format:

image_name,true_label,predicted_label

img001.jpg,rodent,rodent

img002.jpg,rodent,background

Folder Structure

The project has the following directory structure:


rodent-detection/

├── data/

│ ├── images/ # Image files used for training and inference

│ └── instances_default.json # COCO annotations for training

├── models/

│ └── trained_model/ # Directory to store trained model checkpoints

├── outputs/ # Directory to save images and results after inference

├── build.py # Training script

├── eval.py # Inference script

├── utils.py # Utility functions (e.g., collate_fn)

├── requirements.txt # List of Python dependencies

└── README.md # This file

### Dataset

The dataset used for this project is based on rodent detection, formatted in the COCO dataset format with annotations in instances_default.json. Each image in the dataset contains labeled bounding boxes corresponding to the rodent(s) in the image.

  

images/ contains the image files.

instances_default.json contains the annotations in COCO format, including bounding boxes and class labels.

Ensure your dataset is structured similarly before running the scripts.

  

#### Model Details

This project uses Faster R-CNN with a ResNet-50 backbone for object detection. The model is trained to detect rodents (or background). You can customize the model to detect different objects by modifying the annotations and the number of classes.

  

#### Model Training

The model is initialized with pretrained weights from COCO and fine-tuned on the rodent dataset.

The training script (build.py) handles the dataset preparation, model training, and saving checkpoints.

#### Inference

The trained model is loaded from the specified checkpoint, and inference is run on images.

Bounding boxes for detected objects are drawn on the images and saved to the outputs/ folder.

A CSV file with the predicted results is saved alongside the images.

## Notes

The training script (build.py) may take a considerable amount of time depending on the dataset size and computational resources (GPU is recommended).

Ensure that the dataset is formatted correctly and that the paths in the scripts point to the correct directories for your environment.

The model currently supports binary classification: background and rodent. You can modify the code to support multi-class detection if needed.

## Licenses

This project is licensed under the MIT License. See the LICENSE file for details.