import torch
import torchvision
from torchvision import transforms
import json
import os
from PIL import Image
import utils

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

# Main function for inference
def main():
    # Define paths
    annotations_file = 'data/instances_default.json'
    image_dir = 'data/images'
    model_path = 'model_epoch_9.pth'  # Load the final trained model
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
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Run inference on all images in the dataset
    for images, file_names in data_loader:
        images = [image.to(device) for image in images]
        
        with torch.no_grad():
            prediction = model(images)
        
        # Print or save the predictions
        for i, img_name in enumerate(file_names):
            print(f"Predictions for {img_name}: {prediction[i]}")

if __name__ == "__main__":
    main()
