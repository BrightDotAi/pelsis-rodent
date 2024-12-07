import torch
import torchvision
from torchvision import transforms
import json
import os
from PIL import Image
import utils
import cv2
import pandas as pd
import numpy as np

# RodentDataset Class for inference (same as in build.py)
class RodentDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Load the annotations file (for inference, we don't need the annotations)
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        if 'images' not in self.annotations:
            raise KeyError("The COCO annotations file must contain an 'images' key.")
        
        self.images = self.annotations['images']
    
    def __getitem__(self, idx):
        # Get the image filename and load the image
        img_info = self.images[idx]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Apply transformations (if any)
        if self.transforms:
            image = self.transforms(image)

        return image, img_info['file_name']

    def __len__(self):
        return len(self.images)

# Load the trained model for inference
def create_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 classes (background and rodent)
    return model

# Draw Bounding Boxes and Save Image
def draw_boxes(image, boxes, labels, scores, output_path):
    """
    Draw the bounding boxes and labels on the image and save it.
    """
    image = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    image = (image * 255).astype(np.uint8)  # Convert to uint8 (0-255)

    # Ensure image is in BGR format (OpenCV uses BGR by default)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        box = box.astype(int)  # Convert to integers using numpy
        color = (0, 255, 0)  # Green color for boxes
        thickness = 2
        # Draw rectangle around detected object
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
        # Draw label and score
        cv2.putText(image, f'{label}: {score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)  
    
# Main function for inference
def main():
    # Define paths
    annotations_file = 'data/instances_default.json'
    image_dir = 'data/images'
    model_path = 'models/trained_model/model_epoch_9.pth'  # Load the final trained model
    output_dir = 'outputs'  # Directory for output images and CSV
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Data augmentation for inference (only normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Prepare the dataset and dataloaders
    dataset = RodentDataset(annotations_file, image_dir, transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn)
    
    # Initialize the model
    model = create_model()
    
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Prepare CSV for storing results
    results = []

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on all images in the dataset
    for images, file_names in data_loader:
        images = [image.to(device) for image in images]
        
        with torch.no_grad():
            prediction = model(images)
        
        # Process each prediction
        for i, img_name in enumerate(file_names):
            true_labels = prediction[i]['labels'].cpu().numpy()
            predicted_labels = prediction[i]['labels'].cpu().numpy()
            predicted_scores = prediction[i]['scores'].cpu().numpy()
            predicted_boxes = prediction[i]['boxes'].cpu().numpy()

            # Draw and save the image with bounding boxes
            output_image_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_output.jpg")
            draw_boxes(images[i], predicted_boxes, predicted_labels, predicted_scores, output_image_path)

            # Store results for CSV
            for true_label, predicted_label in zip(true_labels, predicted_labels):
                results.append([img_name, true_label, predicted_label])

    # Write results to CSV
    output_csv = os.path.join(output_dir, 'results.csv')
    df = pd.DataFrame(results, columns=['Image Name', 'True Label', 'Predicted Label'])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
