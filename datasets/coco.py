import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from .transforms import get_transform



class CoCo:
    def __init__(self, cfg, dataType='train2017', annType='captions', is_train=True):
        
        if is_train:
            root = cfg.dataset.params.train_path
        else:
            root = cfg.dataset.params.val_path
        
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        self.transform = get_transform(cfg, is_train)
        
        if cfg.experiment.max_train_examples < len(self.imgids):
            self.imgids = self.imgids[:cfg.experiment.max_train_examples]
            
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, ann     
        
    def __len__(self):
        return len(self.imgids)