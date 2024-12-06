import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
THRESHOLD = 0.5
RODENT_CLASSES = [17, 18]  # COCO class IDs for mouse and rat


def load_model():
    """
    Load the pretrained TensorFlow Hub model.

    Returns:
        Model: The loaded model.
    """
    return hub.load(MODEL_URL)


def classify_image(model, image_tensor):
    """
    Run inference on a single image tensor.

    Args:
        model: The loaded model.
        image_tensor: Preprocessed image tensor.

    Returns:
        str: 'Rodent' if detected, otherwise 'No Rodent'.
    """
    output_dict = model(image_tensor)
    detection_classes = output_dict["detection_classes"][0].numpy().astype(np.int64)
    detection_scores = output_dict["detection_scores"][0].numpy()

    for i in range(len(detection_classes)):
        if detection_scores[i] > THRESHOLD and detection_classes[i] in RODENT_CLASSES:
            return "Rodent"
    return "No Rodent"
