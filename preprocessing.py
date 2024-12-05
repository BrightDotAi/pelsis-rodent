from PIL import Image
import numpy as np
import os


def preprocess_image(image_path, target_size=(320, 320)):
    """
    Preprocess a single grayscale image to be compatible with the model.

    Args:
        image_path (str): Path to the image.
        target_size (tuple): Desired image size (width, height).

    Returns:
        tuple: Original PIL image, preprocessed image tensor.
    """
    # Load and convert the grayscale image
    image = Image.open(image_path).convert("L")
    image_resized = image.resize(target_size)
    image_array = np.asarray(image_resized) / 255.0

    # Convert to 3 channels
    image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image, image_array


def load_images_from_folder(folder_path):
    """
    Get paths to all images in a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: List of image file paths.
    """
    return [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]
