from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import scipy
import scipy.io
import os.path

import numpy as np

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name) #.replace('.jpg', label_suffix)

class CDDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        
        self.res = resolution
        self.data_len = data_len
        self.split = split

        self.root_dir = dataroot
        self.split = split  #train | val | test
        
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        
        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A   = Image.open(A_path).convert("RGB")
        img_B   = Image.open(B_path).convert("RGB")
        
        L_path  = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_lbl = Image.open(L_path).convert("RGB")
        
        img_A   = Util.transform_augment_cd(img_A, split=self.split, min_max=(-1, 1))
        img_B   = Util.transform_augment_cd(img_B, split=self.split, min_max=(-1, 1))
        img_lbl = Util.transform_augment_cd(img_lbl, split=self.split, min_max=(0, 1))
        if img_lbl.dim() > 2:
            img_lbl = img_lbl[0]
        
        return {'A': img_A, 'B': img_B, 'L': img_lbl, 'Index': index}
