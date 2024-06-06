import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import time

# from models import *
from adp_r18 import *
from spaware_models import *
from dataloader import PlanetData



def resnet18(num_classes = 1000, normalize = False, use_means = False):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, normalize = normalize, use_means = use_means)



def apply_transforms(stats):
    """Define transformations for training and validation data."""
    transforms_dict = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["global"]["mean"], stats["global"]["std"])
        ])
    }
    return transforms_dict


def construct_model(model_name, norm_coords = False, use_means = False):
    if model_name == "DeepAll":
        model = models.resnet18(pretrained = True)
        model.fc = torch.nn.Linear(512, 1)
    elif model_name == "SpA":
        model = FTSelector()
    elif model_name == "GeoConv":
        model = resnet18(num_classes=1, normalize = norm_coords, use_means = use_means)
    elif model_name == "FC":
        model = MainModel(512, 512, 1)
    return model


def save_model(folder_name, epoch, state_dict, criterion, optimizer, scheduler, best = False):
    
    if best:
        fname = f"{folder_name}/model_epoch{epoch}.torch"
    else:
        fname = f"{folder_name}/most_recent_epoch{epoch}.torch"  
        
    # Save the most current epoch
    torch.save({
                'epoch': epoch,
                'loss': criterion,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, fname)   