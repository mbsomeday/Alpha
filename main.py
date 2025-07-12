import argparse, os
from torch.utils.data import DataLoader

from utils.gen_fade_images import gen_fade_images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_weights_path', type=str,
                        default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
    parser.add_argument('--txt_name', type=str, default='augmentation_train.txt')
    parser.add_argument('--img_save_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\fade_aug_train')

    args = parser.parse_args()
    return args


args = get_args()

func = gen_fade_images(args)
func.calc_cam()










































