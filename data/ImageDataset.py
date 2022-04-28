from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import scipy
import scipy.io

import numpy as np

class ImageDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_len = data_len
        self.split = split

        self.path = Util.get_paths_from_images(dataroot)
            
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.open(self.path[index]).convert("RGB")

        img = Util.transform_augment(img, split=self.split, min_max=(-1, 1), res=self.res)
            
        return {'img': img,  'Index': index}
