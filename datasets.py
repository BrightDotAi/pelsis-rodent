# datasets.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.images = []  # List to store image file paths
        self.annotations = []  # List to store annotations
        self.load_annotations()
        
    def load_annotations(self):
        # Load the annotations from the provided json file
        with open(self.annotation_file) as f:
            self.data = json.load(f)
        
        for image_info in self.data["images"]:
            image_path = os.path.join(self.image_dir, image_info["file_name"])
            self.images.append(image_path)
            
        for annotation in self.data["annotations"]:
            self.annotations.append(annotation)

    def __getitem__(self, idx):
        # Load image and target (annotation)
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Create target dict (bounding boxes, labels, etc.)
        target = {
            "boxes": torch.tensor(self.annotations[idx]["bbox"], dtype=torch.float32),
            "labels": torch.tensor(self.annotations[idx]["category_id"], dtype=torch.int64)
        }
        
        return image, target

    def __len__(self):
        return len(self.images)
