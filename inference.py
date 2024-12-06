import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2
import os
import csv

# Paths
PROJECT_DIR = "/Users/ashish/Repos/pelsis-rodent"
MODEL_DIR = os.path.join(PROJECT_DIR, "models/trained_model")
LABEL_MAP_PATH = os.path.join(PROJECT_DIR, "annotations/label_map.pbtxt")
IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "inference_results.csv")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

# Load the trained model
print("Loading model...")
detect_fn = tf.saved_model.load(os.path.join(MODEL_DIR, "saved_model"))
print("Model loaded successfully.")

# Prepare CSV for saving results
with open(CSV_OUTPUT_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image File", "Prediction"])

    # Process each image in the directory
    for image_file in os.listdir(IMAGE_DIR):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(IMAGE_DIR, image_file)
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Perform inference
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            # Extract the highest-confidence detection
            scores = np.squeeze(detections["detection_scores"])
            classes = np.squeeze(detections["detection_classes"]).astype(np.int32)

            # Use the highest-confidence detection for labeling
            if len(scores) > 0 and scores[0] > 0.5:  # Adjust threshold if needed
                predicted_class = category_index[classes[0]]["name"]
            else:
                predicted_class = "no_rodent"

            # Write to CSV
            csv_writer.writerow([image_file, predicted_class])

            # Visualize detections and save annotated image
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(detections["detection_boxes"]),
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5,
            )
            output_path = os.path.join(OUTPUT_DIR, image_file)
            output_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_image)
            print(f"Processed: {image_file}, Prediction: {predicted_class}")

print(f"Inference results saved to: {CSV_OUTPUT_PATH}")
