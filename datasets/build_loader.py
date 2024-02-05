import torch
from .coco import CoCo
from .transforms import get_transform
from torchvision.datasets import ImageFolder


def build_loader(cfg):
	if cfg.dataset.name == "coco":
		train_ds = CoCo(cfg, dataType='train2017', annType='captions', is_train=True)
		
		if cfg.dataset.params.train_test_split:
			train_size = int(cfg.dataset.params.train_test_split * len(train_ds))
			val_size = len(train_ds) - train_size
			train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

		else:
			val_ds = CoCo(cfg, dataType='val2017', annType='captions', is_train=False)

	if cfg.dataset.name == "imagenet":
		train_ds = ImageFolder(root=cfg.dataset.params.train_path, transform=get_transform(cfg))
		if cfg.dataset.params.train_test_split:
			train_size = int(cfg.dataset.params.train_test_split * len(train_ds))
			val_size = len(train_ds) - train_size
			train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
		else:
			assert False, "Train test split is required for imagenet dataset"


	train_dl = torch.utils.data.DataLoader(train_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers)  
	val_dl = torch.utils.data.DataLoader(val_ds,
											batch_size=cfg.dataset.params.batch_size, 
											shuffle=cfg.dataset.params.shuffle, 
											num_workers=cfg.dataset.params.num_workers) 
	
	return (train_dl, val_dl)


			
			
 