'''
    临时的dataset

'''

import os, sys
import random

import matplotlib.pyplot as plt

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as F

from configs.paths_dict import PATHS


class temp_dataset(Dataset):
    def __init__(self, base_dir, label=2):
        '''
            base_dir下面有pedestrian, nonPedestrian
        '''
        class_dir_list = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
        self.image_list = []
        self.label_list = []
        for cls_dir in class_dir_list:
            images = [os.path.join(cls_dir, img_name) for img_name in os.listdir(cls_dir)]
            self.image_list.extend(images)
            self.label_list.extend([label] * len(images))
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)

        label = self.label_list[idx]

        return image, label




if __name__ == '__main__':
    base_dir = r'D:\my_phd\dataset\Stage6\stage6_bdd100k\fade_aug_train'
    get_dataset = temp_dataset(base_dir, label=2)
    get_loader = DataLoader(get_dataset, batch_size=2)

    for image, label in get_loader:
        print(label)
        print(image.shape)

        break




































