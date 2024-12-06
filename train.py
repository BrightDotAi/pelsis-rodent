import os
import tensorflow as tf
from object_detection import model_lib_v2

# Paths
PROJECT_DIR = "/Users/ashish/Repos/pelsis-rodent"
PIPELINE_CONFIG_PATH = os.path.join(PROJECT_DIR, "pipeline.config")
MODEL_DIR = os.path.join(PROJECT_DIR, "models/trained_model")

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_model():
    print("Starting training...")
    model_lib_v2.train_loop(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        train_steps=None,
        checkpoint_every_n=1000,
        record_summaries=True
    )
    print("Training complete.")

if __name__ == "__main__":
    train_model()
