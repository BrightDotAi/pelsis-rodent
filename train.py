import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection import model_lib_v2
from object_detection.utils import model_util

# Define paths
pipeline_config_path = "/Users/ashish/Repos/pelsis-rodent/pipeline.config"
model_dir = "/Users/ashish/Repos/pelsis-rodent/output/trained_model"

# Load the pipeline config file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

# Read the pipeline config file and parse it
with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Check if the model directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Configure the pipeline to create the model
configs = config_util.create_configs_from_pipeline_proto(pipeline_config)

# Setup for model training
# Ensure that the checkpoint directory is correct
checkpoint_dir = model_dir + "/checkpoint"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialize the model for training
model_config = configs['model']
model = model_util.create_model(model_config)

# Load the checkpoint path for the pre-trained model (if applicable)
fine_tune_checkpoint = "/Users/ashish/Repos/pelsis-rodent/pretrained_model/checkpoint/ckpt-0"

# Run the training
train_input_config = configs['train_input_config']
eval_input_config = configs['eval_input_config']

# Set up the model for training with the pipeline config
train_config = configs['train_config']

# Initiate the training process
model_lib_v2.train_loop(
    pipeline_config_path=pipeline_config_path,
    model_dir=model_dir,
    train_steps=train_config.num_steps,
    checkpoint_dir=checkpoint_dir,
    fine_tune_checkpoint=fine_tune_checkpoint
)

print("Training complete. Model saved in directory:", model_dir)
