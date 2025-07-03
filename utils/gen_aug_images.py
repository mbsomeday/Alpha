from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os, argparse
from tqdm import tqdm

from data.dataset import my_dataset
from utils.utils import save_image_tensor



def image_aug(base_dir):
    get_dataset = my_dataset(ds_name_list=['D4'], path_key='Stage6_org', txt_name='test.txt')
    get_loader = DataLoader(get_dataset, batch_size=1, shuffle=False)

    plt_transform = transforms.ToPILImage()
    bright_transform = transforms.transforms.ColorJitter(brightness=0.2)

    for idx, data_dict in enumerate(tqdm(get_loader)):
        print(data_dict.keys())

        images = data_dict['image']
        image_paths = data_dict['img_path']
        path_contenst = image_paths[0].split(os.sep)
        image_name = path_contenst[-1]

        # img_flip = F.hflip(images)
        # flip_name = image_name.split('.')[0] + '_flip.jpg'
        # img_flip_save_path = os.path.join(base_dir, 'augmentation_train', path_contenst[-2], flip_name)
        # save_image_tensor(img_flip, img_flip_save_path)

        img_bright = bright_transform(images)
        bright_name = image_name.split('.')[0] + '_bright.jpg'
        img_bright_save_path = os.path.join(base_dir, 'augmentation_train', path_contenst[-2], bright_name)
        save_image_tensor(img_bright, img_bright_save_path)

        org_image_save_path = os.path.join(base_dir, 'augmentation_train', path_contenst[-2], image_paths[-1])
        save_image_tensor(images, org_image_save_path)


        # plt.figure()
        # plt.subplot(131)
        # plt.title('org')
        # plt.imshow(plt_transform(images[0]))
        # plt.subplot(132)
        # plt.title('flip')
        # plt.imshow(plt_transform(img_flip[0]))
        # plt.subplot(133)
        # plt.title('brightness')
        # plt.imshow(plt_transform(img_bright[0]))
        # plt.show()

        # break


























