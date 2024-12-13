import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import os
import random
import numpy as np
def all_elements_equal(arr, v):
    return np.all(arr == v) 

def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

MODEL_NAME =  'bert-base-uncased' 
dataset_name = "glue"

# Load different datasets from the GLUE benchmark

glue_datasets = {
    "mnli":  load_dataset( dataset_name, 'mnli'),
    "qqp":  load_dataset( dataset_name, 'qqp'),
    "qnli":  load_dataset( dataset_name, 'qnli'),
    "sst2":  load_dataset( dataset_name, 'sst2'),
    #"cola":  load_dataset( dataset_name, "cola"),
    #"stsb":  load_dataset( dataset_name, "stsb"),
    #"mrpc":  load_dataset( dataset_name, "mrpc"),
    "rte":  load_dataset( dataset_name, 'rte')
}


datasets_num_labels = {
    "mnli": 3,
    "qqp": 2,
    "qnli": 2,
    "sst2": 2,
    #"cola": 2,
    #"mrpc": 2,
    "rte": 2
}

def glue_data():
    return  glue_datasets, datasets_num_labels

def generate_dataloader_nlp(args=None, name="train", dataset =None):
    use_cuda=False
    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=args.rank)

    if name != "train":
        batch_size = args.batch_size_eval
    else:
        batch_size = args.batch_size


    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        sampler=train_sampler,
                        **kwargs)
    return dataloader


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

