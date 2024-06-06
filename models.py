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




class LocationEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10000,
                                      embedding_dim=embedding_dim)
    
    def forward(self, location_ids):
        return self.embedding(location_ids)


class WeightPredictor(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.layer2 = nn.Linear(2, output_dim)
        self.batchnorm2 = nn.BatchNorm1d(output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, location_embedding):
        location_embedding = self.layer2(location_embedding)
        location_embedding = self.batchnorm2(location_embedding)
        location_embedding = self.relu(location_embedding)
        return location_embedding

    
import torch
import torch.nn as nn
import torchvision.models as models

class MainModel(nn.Module):
    def __init__(self, in_features, out_features, num_classes):
        super(MainModel, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        self.resnet18.fc = nn.Identity()
        self.feature_norm = nn.BatchNorm1d(out_features)
        self.dynamic_fc = nn.Linear(out_features, num_classes)
        self.weight_predictor = WeightPredictor(256, 512)
        self.weight_norm = nn.BatchNorm1d(out_features * num_classes)
        self.output_dim = num_classes
        self.input_dim = in_features
        
    def forward(self, x, coords):
        x = self.resnet18(x)
        x = self.feature_norm(x)
        fc_weights = self.weight_predictor(coords) * .0001
        fc_weights = self.weight_norm(fc_weights.view(-1, self.output_dim * self.input_dim))
        fc_weights = fc_weights.view(-1, self.output_dim, self.input_dim)
        x = x.unsqueeze(2)
        x = torch.bmm(fc_weights, x).squeeze(2)
        return x
