from torchvision import transforms as T
from PIL import Image


def pair(x):
    return x, x

def get_transform(cfg, is_train=True):
    resize = cfg.dataset.preprocessing.resolution
    resize = pair(resize)
    t = []
    t.append(T.Resize(resize, interpolation=Image.BICUBIC))
    if is_train:
        if cfg.dataset.preprocessing.random_crop:
            t.append(T.RandomCrop(resize))
        if cfg.dataset.preprocessing.random_flip:
            t.append(T.RandomHorizontalFlip(p=0.5))
        if cfg.dataset.preprocessing.center_crop:
            t.append(T.CenterCrop(resize))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
    
    return T.Compose(t)