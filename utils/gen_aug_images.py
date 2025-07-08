from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os, argparse, random
from tqdm import tqdm
from PIL import Image

from data.dataset import my_dataset
from utils import save_image_tensor


class random_aimage_aug():
    '''
        将augmentation保存到save_dir中
    '''
    def __init__(self, ds_name_list, path_key, save_dir):
        get_dataset = my_dataset(ds_name_list=ds_name_list, path_key=path_key, txt_name='train.txt')
        self.get_loader = DataLoader(get_dataset, batch_size=1, shuffle=False)

        self.save_dir = save_dir
        self.ped_dir = os.path.join(save_dir, 'pedestrian')
        self.nonPed_dir = os.path.join(save_dir, 'nonPedestrian')
        if not os.path.exists(self.ped_dir):
            os.mkdir(self.ped_dir)
        if not os.path.exists(self.nonPed_dir):
            os.mkdir(self.nonPed_dir)

        self.aug_list = [self.hflip, self.rotate, self.jittor, self.gaussian]
        self.aug_name = ['hflip', 'rotate', 'jittor', 'gaussian']
        self.num_list = list(range(len(self.aug_list)))

    def hflip(self, img):
        return F.hflip(img)

    def rotate(self, img):
        angle = random.randint(-10, 10)
        return F.rotate(img, angle)

    def jittor(self, img):
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        return color_jitter(img)

    def gaussian(self, img):
        sigma = random.uniform(0.1, 1.0)
        img = F.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])
        return img

    # def posterize(self, img):
    #     bits = random.randint(3, 6)
    #     img = F.posterize(img, bits)
    #     return img

    def __call__(self):

        for data_dict in tqdm(self.get_loader):
            # print(data_dict.keys())

            # 随机选择一种augmentation
            rand_num = random.choice(self.num_list)
            aug_method = self.aug_list[rand_num]
            aug_name = self.aug_name[rand_num]

            # 读取原始图片
            org_image = data_dict['image']
            org_img_name = data_dict['img_name'][0]
            org_img_path = data_dict['img_path'][0]
            cls_name = org_img_path.split(os.sep)[-2]

            # 保存原始图片
            save_org_path = os.path.join(self.save_dir, cls_name, org_img_name)
            save_image_tensor(org_image, save_org_path)

            # 保存 aug 图片
            save_aug_name = os.path.splitext(org_img_name)[0] + '_' + aug_name + os.path.splitext(org_img_name)[-1]
            save_aug_path = os.path.join(self.save_dir, cls_name, save_aug_name)
            aug_image = aug_method(org_image)
            save_image_tensor(aug_image, save_aug_path)

            break




def image_aug(base_dir):
    get_dataset = my_dataset(ds_name_list=['D4'], path_key='Stage6_org', txt_name='train.txt')
    get_loader = DataLoader(get_dataset, batch_size=1, shuffle=False)

    plt_transform = transforms.ToPILImage()
    bright_transform = transforms.transforms.ColorJitter(brightness=0.2)

    for idx, data_dict in enumerate(tqdm(get_loader)):
        # print(data_dict.keys())

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

        org_image_save_path = os.path.join(base_dir, 'augmentation_train', path_contenst[-2], image_name)
        save_image_tensor(images, org_image_save_path)
        # print(org_image_save_path)



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
        # break

if __name__ == '__main__':
    # img_path = r'D:\my_phd\on_git\DatasetBias\data\img.jpg'
    # img = Image.open(img_path).convert('RGB')

    ww_rand = random_aimage_aug(ds_name_list=['D4'], path_key='Stage6_org', save_dir=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\augmentation_train')
    ww_rand()
























