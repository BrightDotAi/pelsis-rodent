import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse

# Function to preprocess the query image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to classify the query image using TensorFlow (.h5) model
def classify_with_tf(model_path, img_path):
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Run inference
    prediction = model.predict(img_array)
    
    # Interpret the prediction
    if prediction[0] >= 0.5:
        return "rodent"
    else:
        return "not rodent"

# Function to classify the query image using TensorFlow Lite (.tflite) model
def classify_with_tflite(model_path, img_path):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess the image
    img_array = preprocess_image(img_path)
    img_array = img_array.astype(np.float32)  # Ensure the data type is float32
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Interpret the prediction
    if prediction[0] >= 0.5:
        return "rodent"
    else:
        return "not rodent"

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Classify an image as 'rodent' or 'not rodent' using a pre-trained model.",
        epilog="Example usage:\n"
              "  python classify_image.py tf path/to/your/query_image.jpg\n"
              "  python classify_image.py tflite path/to/your/query_image.jpg",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_type",
        choices=["tf", "tflite"],
        help="Type of model to use:\n"
             "  'tf' for TensorFlow (.h5) model\n"
             "  'tflite' for TensorFlow Lite (.tflite) model"
    )
    parser.add_argument(
        "image_path",
        help="Path to the query image file (e.g., 'path/to/your/query_image.jpg')"
    )
    args = parser.parse_args()
    
    # Define model paths
    tf_model_path = "models/rodent_classifier.h5"
    tflite_model_path = "models/rodent_classifier.tflite"
    
    # Classify the image based on the model type
    if args.model_type == "tf":
        result = classify_with_tf(tf_model_path, args.image_path)
    elif args.model_type == "tflite":
        result = classify_with_tflite(tflite_model_path, args.image_path)
    else:
        raise ValueError("Invalid model type. Choose 'tf' or 'tflite'.")
    
    # Print the result
    print(f"The image is classified as: {result}")


if __name__ == "__main__":
    main()