""" process the negative sample images """

import os
import cv2
import random
from tqdm import tqdm

# Define paths
input_dir = "data/not_rodent_raw"  # Directory containing raw images
output_dir = "data/not_rodent"     # Directory to save preprocessed images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Randomly select 100 images
selected_images = random.sample(image_files, 100)

# Process and save images
for idx, image_file in enumerate(tqdm(selected_images, desc="Processing Images")):
    # Load image
    img_path = os.path.join(input_dir, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Resize image to 324x324
    img_resized = cv2.resize(img, (324, 324))

    # Save the preprocessed image with sequential naming
    output_path = os.path.join(output_dir, f"img{idx + 1}.png")
    cv2.imwrite(output_path, img_resized)

print(f"Preprocessed {len(selected_images)} images and saved to {output_dir}.")