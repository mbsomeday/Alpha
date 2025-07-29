import argparse, os

import torch

from utils.gen_fade_images import gen_fade_images

'''
用于图片生成
'''
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
#     parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
#     parser.add_argument('--txt_name_list', nargs='+', default=['augmentation_train.txt'])
#     parser.add_argument('--img_save_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\fade_aug_train')
#
#     args = parser.parse_args()
#     return args
#
#
# args = get_args()
#
# func = gen_fade_images(args)
# func.calc_cam()


'''
dataset classifier
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from data.temp_dataset import temp_dataset
from models.EfficientNet import efficientNetB0
from utils.utils import load_model, DEVICE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\fade_aug_train')
    parser.add_argument('--label', type=int, default=2)

    parser.add_argument('--ds_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')


    args = parser.parse_args()
    return args


args = get_args()

base_dir = args.base_dir
label = args.label
ds_weights = args.ds_weights


ds_dataset = temp_dataset(base_dir, label=2)
ds_loader = DataLoader(ds_dataset, batch_size=2)

ds_model = efficientNetB0(num_class=3)
ds_model = load_model(ds_model, ds_weights)
ds_model.eval()

y_label = []
y_pred = []
correct_num = 0
with torch.no_grad():
    for idx, image, label in enumerate(tqdm(ds_loader)):
        images = image.to(DEVICE)
        ds_label = label.to(DEVICE)

        logits = ds_model(images)
        probs = torch.argmax(logits, 1)

        y_label.extend(ds_label.cpu().numpy())
        y_pred.extend(probs.cpu().numpy())

        correct_num += (ds_label == probs).sum()

    cm = confusion_matrix(y_label, y_pred)
    print(f'Testing cm:\n {cm}')

    ds_accuracy = correct_num / len(ds_dataset)
    print(f'准确率为：{ds_accuracy}')

    bc = balanced_accuracy_score(y_label, y_pred)
    print(f'balanced accuracy为：{bc}')









































