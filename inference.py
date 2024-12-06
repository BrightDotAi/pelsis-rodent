# inference.py
import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
from models import get_model
from utils import save_predictions_to_csv

def infer(model, image_path, device):
    """
    Perform inference on a single image.
    """
    image = F.to_tensor(Image.open(image_path)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    return outputs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('outputs/model_epoch_9.pth'))
    model.to(device)

    image_dir = 'data/images/'
    output_csv = 'outputs/predictions.csv'
    results = {}

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            outputs = infer(model, image_path, device)
            # Example: Take the first prediction label with the highest score
            labels = outputs[0]['labels'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            if len(scores) > 0 and scores[0] > 0.5:
                results[image_file] = "rodent" if labels[0] == 2 else "no_rodent"
            else:
                results[image_file] = "no_detection"

    save_predictions_to_csv(results, output_csv)

if __name__ == "__main__":
    main()
