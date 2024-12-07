import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
from PIL import Image
import torch.optim as optim
from engine import train_one_epoch, evaluate
import utils

# Custom transform to handle both image and target (for COCO format)
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        # Apply the image transformation
        image = self.transform(image)
        # Return both image and target
        return image, target

# RodentDataset Class for handling the COCO annotations
class RodentDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Load the annotations file
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        if 'images' not in self.annotations:
            raise KeyError("The COCO annotations file must contain an 'images' key.")
        
        self.images = self.annotations['images']
        self.annotations_data = self.annotations['annotations']

    def __getitem__(self, idx):
        # Get the image filename and load the image
        img_info = self.images[idx]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Prepare targets (annotations)
        target = {}
        annotations = [anno for anno in self.annotations_data if anno['image_id'] == img_info['id']]
        
        # Extract boxes and labels from annotations for this image
        boxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']
            # Ensure the bounding box has positive width and height
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue  # Skip invalid bounding boxes
            
            # Ensure the top-left corner is smaller than the bottom-right corner
            x2 = x + w
            y2 = y + h
            if x > x2:
                x, x2 = x2, x
            if y > y2:
                y, y2 = y2, y
            
            # Append corrected box and label
            boxes.append([x, y, x2, y2])
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            # Skip this image if there are no valid bounding boxes
            return self.__getitem__((idx + 1) % len(self.images))

        # Convert to tensor
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations (if any)
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)


# Define model setup
def create_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 classes (background and rodent)
    return model

# Main training function
def main():
    # Define paths
    annotations_file = 'data/instances_default.json'
    image_dir = 'data/images'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Data augmentation for training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create the custom transform that handles both image and target
    custom_transform = CustomTransform(transform)
    
    # Prepare the dataset and dataloaders
    dataset = RodentDataset(annotations_file, image_dir, transforms=custom_transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    
    # Initialize the model
    model = create_model()
    model.to(device)
    
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Number of epochs
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Save the model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        
    print("Training complete!")

if __name__ == "__main__":
    main()
