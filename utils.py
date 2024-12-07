import torch
import time
from collections import deque
from typing import List, Dict
from torch.utils.data import DataLoader
import numpy as np

def reduce_dict(input_dict: Dict):
    # Reduce loss dictionary to a single value for each entry
    world_size = 1
    reduced_dict = {}
    for key, value in input_dict.items():
        reduced_dict[key] = value / world_size
    return reduced_dict

class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(name)
            self.meters[name].update(value)

    def log_every(self, iterable, print_freq, header=None):
        # Log every x steps
        if header:
            print(header)
        for i, data in enumerate(iterable):
            yield data
            if i % print_freq == 0:
                self.print_metrics()

    def print_metrics(self):
        # Print metrics to stdout
        entries = [f'{name}: {meter.avg:.3f}' for name, meter in self.meters.items()]
        print(self.delimiter.join(entries))

class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

# Collate function for batching variable-length inputs
def collate_fn(batch):
    return tuple(zip(*batch))

