import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import os
from datasets import RodentDataset
from engine import train_one_epoch, evaluate
from utils import MetricLogger, reduce_dict

def main():
    # Setup paths
    data_dir = 'data'
    image_dir = os.path.join(data_dir, 'images')
    tfrecord_path = os.path.join(data_dir, 'rodent_dataset.tfrecord')
    coco_output_path = 'coco_annotations'  # Folder for COCO annotations
    model_dir = 'models'
    output_model_path = os.path.join(model_dir, 'rodent_model.pth')
    
    # Convert TFRecord to COCO
    print(f"Converting TFRecord to COCO format: {tfrecord_path} -> {coco_output_path}")
    parse_tfrecord_to_coco(tfrecord_path, coco_output_path, image_dir)

    # Load dataset
    dataset = RodentDataset(coco_output_path)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)

    # Load pre-trained model (e.g., Faster R-CNN)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 classes: rodent and no_rodent

    # Setup optimizer and device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        evaluate(model, data_loader, device=device)
        
    # Save the trained model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    main()
