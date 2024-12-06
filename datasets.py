import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class RodentDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Load the annotations file
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Check if 'images' key exists, and assign it
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
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
        
        # Convert to tensor
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        # Generate image_id based on the filename or another method
        image_id = img_info['file_name']

        # Apply transformations (if any)
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
