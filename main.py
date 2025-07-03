import argparse, os
from torch.utils.data import DataLoader

# from utils.gen_fade_images import gen_fade_images
# from data.dataset import my_dataset

from utils.gen_aug_images import image_aug

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str)

    args = parser.parse_args()
    return args


args = get_args()
base_dir = args.base_dir

image_aug(base_dir)








































