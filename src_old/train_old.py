import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


# def fine_tune_model():
#     # Set paths
#     model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
#     model_dir = "output/trained_model"
#     train_record_path = "dataset/rodent_dataset.tfrecord"
#     label_map_path = "dataset/label_map.pbtxt"
#     config_path = "pipeline.config"

#     # Load the pretrained model
#     model = hub.load(model_url)

#     # Update pipeline config for fine-tuning
#     configs = config_util.get_configs_from_pipeline_file(config_path)
#     pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
#     with tf.io.gfile.GFile(config_path, "r") as f:
#         proto_str = f.read()
#         text_format.Merge(proto_str, pipeline_config)

#     # Update the pipeline config for your dataset
#     pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_path]
#     pipeline_config.train_input_reader.label_map_path = label_map_path
#     pipeline_config.train_config.fine_tune_checkpoint = tf.train.latest_checkpoint(model_dir)
#     pipeline_config.train_config.num_steps = 2000  # Adjust based on dataset size
#     pipeline_config.train_config.batch_size = 8

#     config_util.save_pipeline_config(pipeline_config, model_dir)

#     # Train the model
#     print("Starting training...")
#     tf.compat.v1.app.run(main=train_model_main, argv=[
#         "--pipeline_config_path", config_path,
#         "--model_dir", model_dir
#     ])


# if __name__ == "__main__":
#     fine_tune_model()

