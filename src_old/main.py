from preprocessing import preprocess_image, load_images_from_folder
from inference import load_model, classify_image
from utils import write_results_to_csv
import os

INPUT_FOLDER = "input_images"
OUTPUT_FILE = "output.csv"


def main():
    # Load images
    image_paths = load_images_from_folder(INPUT_FOLDER)

    # Load the model
    model = load_model()

    # Process images and classify
    results = []
    for image_path in image_paths:
        _, image_tensor = preprocess_image(image_path)
        label = classify_image(model, image_tensor)
        results.append((image_path, label))

    # Write results to CSV
    write_results_to_csv(OUTPUT_FILE, results)
    print(f"Classification completed. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
