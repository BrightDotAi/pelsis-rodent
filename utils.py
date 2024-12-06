# utils.py
import os
import json
import tensorflow as tf
from PIL import Image
import torch
import torchvision
from collections import deque
import time
import sys
import numpy as np
import torch.distributed as dist

def convert_to_native(data):
    """
    Recursively convert numpy types to native Python types (e.g., np.int64 -> int, np.float64 -> float)
    """
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_native(value) for key, value in data.items()}
    else:
        return data


def parse_tfrecord_to_coco(tfrecord_path, output_json, image_dir):
    """
    Convert TFRecord to COCO-style JSON annotations.
    """
    # Initialize TFRecord reader
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Create an empty list to store parsed data
    data = []

    # Define the feature dictionary for parsing
    def parse_tfrecord_example(proto):
        # Define the features to be extracted from TFRecord
        keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([1], tf.int64),
            'image/width': tf.io.FixedLenFeature([1], tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        # Parse the example
        parsed_example = tf.io.parse_single_example(proto, keys_to_features)
        return parsed_example

    # Parse the TFRecord
    for raw_example in raw_dataset:
        parsed_example = parse_tfrecord_example(raw_example)
        
        image_path = os.path.join(image_dir, parsed_example['image/filename'].numpy().decode())
        width = parsed_example['image/width'].numpy()[0]
        height = parsed_example['image/height'].numpy()[0]
        
        annotations = []
        bboxes = parsed_example['image/object/bbox/xmin'].values
        labels = parsed_example['image/object/class/label'].values
        
        for xmin, ymin, xmax, ymax, label in zip(bboxes[::4], bboxes[1::4], bboxes[2::4], bboxes[3::4], labels):
            annotations.append({
                "bbox": [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height],
                "category_id": int(label.item())  # Convert int64 to Python int
            })

        data.append({
            "image_id": parsed_example['image/filename'].numpy().decode(),
            "file_name": image_path,
            "height": height,
            "width": width,
            "annotations": annotations
        })

    # Convert all data to native Python types
    data = convert_to_native(data)

    # Save the annotations to a JSON file in COCO format
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)


# Function to reduce dictionary of metrics across all workers in distributed training
def reduce_dict(input_dict):
    """
    Reduces the dictionary of metrics by averaging across all processes.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_sum = {}
    all_count = {}

    for key, value in input_dict.items():
        all_sum[key] = value
        all_count[key] = 1

    # Reduce all values
    dist.all_reduce(all_sum)
    dist.all_reduce(all_count)

    for key in all_sum:
        all_sum[key] = all_sum[key] / world_size

    return all_sum

# SmoothedValue is used to calculate moving averages
class SmoothedValue:
    def __init__(self, window_size=20, fmt="{avg:.4f}"):
        self.deque = deque(maxlen=window_size)
        self.fmt = fmt
        self.total = 0.0
        self.count = 0

    def update(self, val):
        self.deque.append(val)
        self.total += val
        self.count += 1

    @property
    def avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(avg=self.avg)

# MetricLogger will keep track of multiple metrics for each training step
class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = SmoothedValue()
            self.meters[name].update(value)

    def __str__(self):
        return self.delimiter.join(
            f"{name} {str(meter)}" for name, meter in self.meters.items()
        )

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header:
            print(header)
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                print(self)

# Collate function to handle different types of input data in the dataset
def collate_fn(batch):
    return tuple(zip(*batch))
