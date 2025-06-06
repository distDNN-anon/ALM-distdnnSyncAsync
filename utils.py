import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import os
import random
import numpy as np

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

preprocess_transform = T.Compose([
                T.Resize(256), 
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),  
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

DATA_DIR = './datasets/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')



def generate_dataloader(args=None, data=TRAIN_DIR, name="train", transform=preprocess_transform):
    if data is None: 
        return None
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    use_cuda=False
    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=args.rank)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                        sampler=train_sampler,
                        **kwargs)
    return dataloader


def accuracy(output, target, topk=(1, 5)):
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
        output (torch.Tensor): Model outputs of shape (batch_size, num_classes).
        target (torch.Tensor): True labels of shape (batch_size).
        topk (tuple): Tuple of k values for which to calculate accuracy.

    Returns:
        list: List of accuracies for each k in topk.
    """
    maxk = max(topk)

    # Get the top-k predictions and their indices
    _, pred = output.topk(maxk, 1, True, True)  # (batch_size, maxk)
    pred = pred.t()  # Transpose to (maxk, batch_size)

    # Compare with the ground truth and get the correct predictions
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, batch_size)

    # Calculate accuracy for each k
    accuracies = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # Sum correct predictions for top-k
        accuracies.append(correct_k) 

    return accuracies


