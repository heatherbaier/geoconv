import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import torch.nn.init as init

import torch
import torch.nn as nn
import torch.nn.functional as F




class LatLongProjection(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 128], output_size=128):
        super(LatLongProjection, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        original_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = nn.Linear(512 + 128, 1)

    def forward(self, img_features, coords_features):
        combined_features = torch.cat((img_features, coords_features), dim = 1)
        out = self.fc(combined_features)
        return out

    
class FTSelector(nn.Module):
    def __init__(self):
        super(FTSelector, self).__init__()
        self.feature_extractor = ModifiedResNet18()
        self.geo_proj = LatLongProjection()

    def forward(self, x, coords):
        img_features = self.feature_extractor.features(x)
        img_features = torch.flatten(img_features, 1)
        coords_features = self.geo_proj(coords)
        pred = self.feature_extractor(img_features, coords_features)
        return pred

