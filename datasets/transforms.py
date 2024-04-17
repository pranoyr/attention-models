from torchvision import transforms as T
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt



def pair(x):
    return x, x

def get_transform(cfg, is_train=True):
    size = cfg.dataset.preprocessing.resolution
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
    if cfg.dataset.preprocessing.mean:
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
                "random_flip": False,
                "center_crop": False,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "scale" : 1.0
            }
        }
    }
    cfg = OmegaConf.create(cfg)
    img = Image.open("data/images/1.jpg")
    transform = get_transform(cfg, is_train=True)
    while True:
        img_transformed = transform(img)
        # img_numpy = img_transformed.view(-1).numpy()
        # show the distribution of the image
        # plt.hist(img_numpy, bins=100)
        # # save the image
        # plt.savefig("hist.jpg")
        

        grid = make_grid(img_transformed, nrow=6, normalize=True, value_range=(-1, 1))
        # save_image(grid, "t.jpg")
        grid = grid.permute(1, 2, 0).detach().cpu().numpy()
        cv2.imshow("img", grid)
        # cv2.imwrite("t.jpg", grid)
        # break
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break