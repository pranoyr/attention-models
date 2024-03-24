import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from .transforms import get_transform


    
class ImageNet:
    def __init__(self, root, split='train', transform=None):
        self.dataset = torchvision.datasets.ImageNet(root=root, split=split)
        self.transform = transform
        self.prefix = ["an image of ", "a picture of "]
        self.transform = get_transform(cfg, is_train)
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, caption
    
    def __len__(self):
        return len(self.dataset)