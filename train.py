import torch
import torchvision
from torch.utils.data import DataLoader
import os
from datasets import RodentDataset
from transforms import ToTensor
from engine import train_one_epoch, evaluate
import utils

def main():
    # Directories and file paths
    image_dir = 'data/images'
    annotation_file = 'data/annotations/instances_default.json'
    output_dir = 'outputs'
    
    # Prepare dataset and dataloaders
    dataset = RodentDataset(annotation_file, image_dir, transforms=ToTensor())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3  # Background + rodent + no_rodent
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    for epoch in range(10):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        evaluate(model, data_loader, device)

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))

if __name__ == "__main__":
    main()
