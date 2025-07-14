# 将上级目录加入 sys.path， 防止命令行运行时找不到包
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



class RandomAug():
    '''
        对训练image进行randomaugmentation
    '''
    def __call__(self, img):

        # 左右翻转
        if random.random() > 0.5:
            img = F.hflip(img)

        # 随机旋转
        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            img = F.rotate(img, angle)

        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            img = color_jitter(img)

        # Random Gaussian blur
        if random.random() > 0.3:
            sigma = random.uniform(0.1, 1.0)
            img = F.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])

        # Random posterize
        if random.random() > 0.1:
            bits = random.randint(3, 6)
            img = F.posterize(img, bits)

        # Convert to tensor and normalize
        img = F.to_tensor(img)
        # img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img




class my_dataset(Dataset):
    def __init__(self, ds_name_list, path_key, txt_name):
        '''
        :param ds_name_list:
        :param path_key: org_dataset
        :param txt_name:
        '''
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key
        for ds_name in ds_name_list:
            self.ds_label_list.append(int(ds_name[1]) - 1)

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.images, self.ped_labels, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ped_labels, ds_labels = [], [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)
            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ped_labels.append(contents[-1])
                ds_labels.append(ds_label)

        return images, ped_labels, ds_labels

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''
        nonPed_num, ped_num = 0, 0
        for item in self.ped_labels:
            if item == '0':
                nonPed_num += 1
            elif item == '1':
                ped_num += 1
        return nonPed_num, ped_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.ped_labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'img_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict



class operated_dsimages(Dataset):
    '''
        读取org image和fade images，仅限augmentation_train
    '''
    def __init__(self, ds_name_list, path_key, operated_dir_name):
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key
        self.operated_dir_name = operated_dir_name
        for ds_name in ds_name_list:
            self.ds_label_list.append(int(ds_name[1]) - 1)
        self.txt_name = 'augmentation_train.txt'
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.org_images, self.operated_images, self.ped_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {self.txt_name}, total {len(self.org_images)} images')

    def __len__(self):
        return len(self.org_images)

    def init_ImagesLabels(self):
        org_images, operated_image, ped_labels = [], [], []
        for ds_idx, ds_name in enumerate(self.ds_name_list):

            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)
            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                org_image_path = os.path.join(ds_dir, contents[0])
                operated_img_path = org_image_path.replace('augmentation_train', self.operated_dir_name)

                org_images.append(org_image_path)
                operated_image.append(operated_img_path)
                ped_labels.append(contents[-1])

        return org_images, operated_image, ped_labels

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''
        nonPed_num, ped_num = 0, 0
        for item in self.ped_labels:
            if item == '0':
                nonPed_num += 1
            elif item == '1':
                ped_num += 1
        return nonPed_num, ped_num

    def __getitem__(self, idx):
        org_image_path = self.org_images[idx]
        ope_image_path = self.operated_images[idx]
        ped_label = self.ped_labels[idx]


        # 读取 org_image 和 operated image
        org_image = Image.open(org_image_path).convert('RGB')
        org_image = self.img_transforms(org_image)
        operated_image = Image.open(ope_image_path).convert('RGB')
        operated_image = self.img_transforms(operated_image)
        ped_label = np.array(ped_label).astype(np.int64)

        image_name = org_image_path.split(os.sep)[-1]

        image_dict = {
            'image': org_image,
            'img_name': image_name,
            'img_path': org_image_path,
            'ped_label': ped_label,
            'ope_image': operated_image
        }

        return image_dict






def get_data(ds_name_list, path_key, txt_name, batch_size, shuffle=True):
    get_dataset = my_dataset(ds_name_list, path_key, txt_name)
    get_loader = DataLoader(get_dataset, batch_size=batch_size, shuffle=shuffle)
    return get_dataset, get_loader



class dataset_from_list(Dataset):
    '''
        从txt中读取image path，用于计算cam，不需要返回label
    '''
    def __init__(self, txt_path):
        self.txt_path = txt_path
        print(f'Data loaded from {txt_path}')
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))     # 这是根据ImageNet计算的，可以作为通用的均值和方差
        ])

        self.images = self.init_Images()

    def init_Images(self):
        images = []
        with open(self.txt_path, 'r') as f:
            data = f.readlines()

        for item in data:
            image_path = item.strip()
            images.append(image_path)

        return images
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        image = self.img_transforms(image)
        return image



class dataset_clip(Dataset):
    def __init__(self, ds_name_list, path_key, txt_name):
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key
        for ds_name in ds_name_list:
            self.ds_label_list.append(int(ds_name[1]) - 1)

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ds_labels = [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'clip', 'dataset_txt', self.txt_name)
            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ds_labels.append(ds_label)

        return images, ds_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path)
        image = self.img_transforms(image)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'clip': image,
            'img_name': image_name,
            'img_path': image_path,
            'ds_label': ds_label
        }

        return image_dict



if __name__ == '__main__':
    get_dataset = operated_dsimages(ds_name_list=['D3'], path_key='Stage6_org')
    get_loader = DataLoader(get_dataset, batch_size=4, shuffle=False)

    for idx, data in enumerate(get_loader):
        print(data.keys())

        break























