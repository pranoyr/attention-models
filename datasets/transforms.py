from torchvision import transforms as T
from PIL import Image
from torchvision import transforms


def pair(x):
    return x, x

def get_transform(cfg, is_train=True):
    size = cfg.dataset.preprocessing.resolution
    # resize = pair(resize)
    scale = cfg.dataset.preprocessing.scale
    if not is_train:
        scale = 1.0
 
    resize = pair(int(size/scale))
    t = []
    t.append(T.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR))
    if is_train:
        if cfg.dataset.preprocessing.random_crop:
            t.append(T.RandomCrop(size))
        if cfg.dataset.preprocessing.random_flip:
            t.append(T.RandomHorizontalFlip(p=0.5))
        if cfg.dataset.preprocessing.center_crop:
            t.append(T.CenterCrop(size))
    else:
        t.append(T.CenterCrop(size))
        
    t.append(T.ToTensor())
    t.append(T.Normalize(mean=(cfg.dataset.preprocessing.mean), std=(cfg.dataset.preprocessing.std)))
    
    return T.Compose(t)

if __name__=="__main__":
    import cv2
    import numpy as np
    from omegaconf import DictConfig, ListConfig, OmegaConf
    cfg = {
        "dataset": {
            "preprocessing": {
                "resolution": 256,
                "random_crop": True,
                "random_flip": True,
                "center_crop": False,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "scale" : 0.66
            }
        }
    }
    cfg = OmegaConf.create(cfg)
    img = Image.open("data/images/000000000191.jpg")
    transform = get_transform(cfg, is_train=True)
    while True:
        img_transformed = transform(img)
        img_numpy = img_transformed.permute(1, 2, 0).numpy()
        print(img_numpy.shape)
        cv2.imshow("img", img_numpy)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break