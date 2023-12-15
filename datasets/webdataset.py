import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn
from random import randrange
import os
os.environ["WDS_VERBOSE_CACHE"] = "1"

import webdataset as wds

url = "http://0.0.0.0:8000/dataset.tar"
dataset = wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("jpg","xml")

for i in dataset:
    print(i[1])

# print(dataset)


# for image in dataset:
#     print(image.shape)
#     # plt.imshow(image)
#     # plt.show()
#     # break