import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

from collections import OrderedDict
import random


class PlanetData(Dataset):

    def __init__(self, labels_path, transform = None, sample = 0, postval=False, ts4 = False, valnames = None):
        
        print(sample)

        self.postval = postval
        
        if ts4:
            prefix = "/rapids/notebooks/"
        else:
            prefix = ""
        self.ts4 = ts4

        with open(labels_path, "r") as file:
            self.labels = json.load(file)
        if sample > 0:
            self.image_names = random.sample(list(self.labels.keys()), sample)
        elif valnames is not None:
            self.image_names = valnames
        else:
            self.image_names = list(self.labels.keys())

        print(f"Number of images:", len(self.labels))

        # Load geojson with coordinates
        with open(f"{prefix}/sciclone/geograd/Heather/c1/data/clean/coords.json", "r") as file:
            self.geo_info = json.load(file)
        
        self.transform = transform
        self.phase = "train"

        if postval:
            self.phase = "test"

    def set_stage(self, stage):
        self.phase = stage

    def _load_image(self, image_path):
        if ".ipynb" not in image_path:
            image = Image.fromarray(np.array(Image.open(image_path).convert("RGB"))[0:224, 0:224, :])
            return image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        current_image_path = self.image_names[idx]
        lon, lat = self.geo_info[current_image_path]
        current_label = self.labels[current_image_path]
        if not self.ts4:
            current_image_path = current_image_path[17:]
        current_image = self._load_image(current_image_path)

        if self.transform and self.phase in self.transform:
            current_image = self.transform[self.phase](current_image)

        geo_tensor = torch.tensor([lon, lat], dtype=torch.float)

        if not self.postval:
            return current_image, current_label, geo_tensor
        else:
            return current_image, current_label, geo_tensor, current_image_path
        
        
