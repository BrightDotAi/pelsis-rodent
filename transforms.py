# transforms.py
import torch
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]
        return image, target

# transforms.py

class RandomFlip(object):
    def __call__(self, image, target):
        # Ensure that target["boxes"] is a 2D tensor
        boxes = target["boxes"]
        
        if boxes.ndimension() == 1:
            # If there is only one box, expand it to be a 2D tensor
            boxes = boxes.unsqueeze(0)  # Add an extra dimension
        
        # Flip the bounding box coordinates (xmin, xmax)
        boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        
        # Update the target
        target["boxes"] = boxes
        return image, target
