Visual Wake-Up Word: Rodent Detection
This project is a binary image classification system designed to detect the presence of a "rodent" in grayscale images. It uses a pre-trained model (MobileNetV2) with transfer learning and fine-tuning to classify images as either "rodent" or "not rodent". The model is trained on a dataset of grayscale images and can be deployed on low-powered edge devices using TensorFlow Lite.

Table of Contents
Project Overview

Directory Structure

Installation

Usage

Training the Model

Classifying an Image

Model Details

Results

License

Project Overview
The goal of this project is to build a binary classifier that can detect the presence of a rodent in grayscale images. The system is designed to be lightweight and efficient, making it suitable for deployment on edge devices using TensorFlow Lite.

Key Features
Transfer Learning: Uses a pre-trained MobileNetV2 model for feature extraction.

Fine-Tuning: Fine-tunes the model on a custom dataset of rodent and non-rodent images.

TensorFlow Lite Support: Converts the trained model to TensorFlow Lite for deployment on edge devices.

Command-Line Interface: Provides a user-friendly CLI for training and inference.

Directory Structure
Copy
visual-wake-up-word/
├── data/
│   ├── rodent/                # Images of rodents
│   └── not_rodent/            # Images without rodents
├── models/                    # Saved models
│   ├── rodent_classifier.h5   # TensorFlow model
│   └── rodent_classifier.tflite # TensorFlow Lite model
├── results/                   # Evaluation metrics and graphs
│   ├── evaluation_metrics.txt # Test accuracy, loss, and AUC
│   └── training_history.png   # Training/validation accuracy and loss graphs
├── classify_image.py          # Script to classify a query image
├── train_model.py             # Script to train the model
└── README.md                  # Project documentation
Installation
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/visual-wake-up-word.git
cd visual-wake-up-word
Install Dependencies:
Ensure you have Python 3.7 or higher installed. Then, install the required libraries:

bash
Copy
pip install tensorflow numpy matplotlib
Prepare the Dataset:

Place your rodent images in data/rodent/.

Place your non-rodent images in data/not_rodent/.

Ensure all images are in grayscale and resized to 324x324 pixels.

Usage
Training the Model
To train the model, run the following command:

bash
Copy
python train_model.py
This script will:

Load and preprocess the dataset.

Train the model using transfer learning and fine-tuning.

Save the trained model to models/rodent_classifier.h5.

Convert the model to TensorFlow Lite and save it to models/rodent_classifier.tflite.

Save evaluation metrics and training graphs to the results/ directory.

Classifying an Image
To classify a query image, use the classify_image.py script. It supports both TensorFlow (.h5) and TensorFlow Lite (.tflite) models.

Usage
bash
Copy
python classify_image.py <model_type> <image_path>
Arguments
<model_type>: Type of model to use (tf for TensorFlow or tflite for TensorFlow Lite).

<image_path>: Path to the query image.

Examples
Classify an image using the TensorFlow model:

bash
Copy
python classify_image.py tf path/to/your/query_image.jpg
Classify an image using the TensorFlow Lite model:

bash
Copy
python classify_image.py tflite path/to/your/query_image.jpg
Output
The script will print the classification result:

Copy
The image is classified as: rodent
Model Details
Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet).

Custom Layers:

Global Average Pooling.

Dense layer with 128 units and ReLU activation.

Dropout layer with a rate of 0.5.

Output layer with a single unit and sigmoid activation for binary classification.

Training Process
Transfer Learning:

The base model is frozen, and only the custom layers are trained.

Fine-Tuning:

The top layers of the base model are unfrozen, and the entire model is fine-tuned with a lower learning rate.

Evaluation Metrics
Test Accuracy: Accuracy of the model on the test set.

Test AUC: Area under the ROC curve for the test set.

Results
After training, the following files are saved in the results/ directory:

evaluation_metrics.txt:

Contains test accuracy, loss, and AUC.

training_history.png:

Graphs of training/validation accuracy and loss over epochs.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
TensorFlow and Keras for providing the deep learning framework.

MobileNetV2 for the pre-trained model architecture.

