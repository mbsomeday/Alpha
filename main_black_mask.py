import argparse, torch
import os.path
from tqdm import tqdm
from torchcam.methods.gradient import GradCAM
from torchcam.utils import overlay_mask
from torch.utils.data import DataLoader
from models.EfficientNet import efficientNetB0
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as F_vision
from torchvision.utils import save_image

# from PIL import Image
# import numpy as np
# from torchvision.transforms.functional import normalize, resize, to_pil_image

from data.dataset import my_dataset
# from training_template import DS_Classifier
from utils.utils import load_model


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--txt_name', type=str, default='test.txt')
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--ds_model_obj', type=str, default='models.EfficientNet.efficientNetB0')

    parser.add_argument('--isTrain', action='store_true')

    # test
    parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\打乱labelMap的datasetClassifier\dsClsD1D2D3_210-40-0.32013.pth')    # new map

    # parser.add_argument('--ds_weights_path', type=str, default=r'/kaggle/input/stage6-weights-dscls/dsClsD1D2D3-08-1.09839.pth')

    # parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')    # old map local
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--test_txt_name', type=str, default='test.txt')

    parser.add_argument('--save_dir', type=str, default='/kaggle/working/blackMask_D1')

    opts = parser.parse_args()
    return opts


opts = get_opts()

torch.manual_seed(3)

# data
get_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=opts.txt_name)
get_loader = DataLoader(get_dataset, batch_size=opts.test_batch_size, shuffle=True)

# model
ds_model = efficientNetB0(num_class=3)
ds_model = load_model(ds_model, opts.ds_weights_path).eval()

# 行人分类 model
# ped_model = efficientNetB0(num_class=2)
# ped_model = load_model(ped_model, weights_path=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D1\efficientNetB0_D1_3_Baseline-18-4.63655.pth').eval()     # local

# model 变量
model = ds_model

# cam
cam_extractor = GradCAM(model, target_layer='features.8.2')

# visualiz
plt_transformer = transforms.ToPILImage()
tensor_transformer = transforms.ToTensor()
resize_transformer = transforms.Resize((224, 224))
blur_transformer = transforms.GaussianBlur(kernel_size=49)


def get_camAndMask(preds, logits):
    activation_map = cam_extractor(preds.tolist(), logits)[0]
    resized_acmap = resize_transformer(activation_map)
    max_vals = resized_acmap.amax(dim=(1, 2), keepdim=True)
    thresholds = max_vals * 0.5

    masks = torch.where(resized_acmap <= thresholds, resized_acmap, torch.zeros_like(resized_acmap))
    masks = masks.unsqueeze(1)

    return resized_acmap, masks




def func_1():
    for idx, data_dict in enumerate(tqdm(get_loader)):
        # print(data_dict.keys())
        images = data_dict['image']
        ped_labels = data_dict['ped_label']
        ds_labels = data_dict['ds_label']
        image_names = data_dict['img_name']

        # print(f'********** Ped label:{ped_labels}, ds_labels:{ds_labels} **********')

        logits = model(images)
        preds = torch.argmax(logits, 1)
        # print('-' * 50)
        # print(f'preds: {preds}, \nlogits: {logits}, probs: {F.softmax(logits, 1)}')

        # ---------- 批量化操作 ----------

        # 获取cam和mask
        resized_cams, masks = get_camAndMask(preds, logits)

        # 用mask过滤原图
        masked_images = images * masks
        masked_images = torch.where(masks.bool(), images, torch.zeros_like(images))

        # 将图片保存到新的路径

        for save_idx in range(len(images)):
            cur_masked_image = masked_images[save_idx]
            save_path = os.path.join(opts.save_dir, image_names[save_idx])
            save_image(cur_masked_image, save_path)

        # masked_images = (np.array(images)).copy()
        # masked_image[mask==0] = 0
        # masked_image = plt_transformer(masked_image)

        # # show
        # img_idx = 0
        #
        # # 原图
        # one_image = images[img_idx]
        # plt_image = plt_transformer(one_image)

        # get cam and mask
        # resized_cam, mask = get_camAndMask(preds, logits)

        # # 原图 + cam
        # activation_map = to_pil_image(activation_map.squeeze(0), mode='F')
        # result = overlay_mask(plt_image, activation_map, alpha=0.5)

        # # 用mask过滤图片
        # masked_image = (np.array(plt_image)).copy()
        # masked_image[mask==0] = 0
        # masked_image = plt_transformer(masked_image)

        # masked_tensor = tensor_transformer(masked_image).unsqueeze(0)
        # masked_logits = model(masked_tensor)
        # masked_preds = torch.argmax(masked_logits, 1)
        # print('-' * 50)
        # print(f'black masked preds: {masked_preds}, \nlogits: {masked_logits}, probs: {F.softmax(masked_logits, 1)}')

        # 获取第二次的cam
        # cam_02, mask_02 = get_camAndMask(preds=masked_preds, logits=masked_logits)

        # # 用 mask_02 过滤图片
        # masked_image_02 = (np.array(masked_image)).copy()
        # masked_image_02[mask_02==0] = 0
        # masked_image_02 = plt_transformer(masked_image_02)
        #
        # masked_tensor_02 = tensor_transformer(masked_image_02).unsqueeze(0)
        # masked_logits_02 = model(masked_tensor_02)
        # masked_preds_02 = torch.argmax(masked_logits_02, 1)
        # print('-' * 50)
        # print(f'masked preds 02: {masked_preds_02}, \nlogits: {masked_logits_02}, probs: {F.softmax(masked_logits_02, 1)}')

        # print()

        # # blur masked 区域
        # blurred_image = blur_transformer(images[0])
        #
        # # 用mask过滤 blur 图片
        # maskedBlur_image = (np.array(plt_transformer(blurred_image))).copy()
        # maskedBlur_image[mask==1] = 0
        # maskedBlur_image = plt_transformer(maskedBlur_image)
        #
        # # 图片合并
        # # temp_blend = (np.array(maskedBlur_image)).copy()
        # # temp_blend[mask==0] = 0
        #
        # arr1 = np.array(masked_image).astype(np.float32)
        # arr2 = np.array(maskedBlur_image).astype(np.float32)
        # result_arr = arr1 + arr2
        # result_arr = np.clip(result_arr, 0, 255)
        #
        # # 转换回uint8类型并创建PIL Image
        # result_arr = result_arr.astype(np.uint8)
        # result_img = Image.fromarray(result_arr)
        #
        # # 对合并的image进行ped model检测
        # blend_logits = model(tensor_transformer(result_img).unsqueeze(0))
        # blend_preds = torch.argmax(blend_logits, 1)
        # print('-' * 50)
        # print(f'blur preds: {blend_preds}, \nlogits: {blend_logits}, probs: {F.softmax(blend_logits, 1)}')
        #
        # show

        # plt.figure()
        # plt.subplot(2, 4, 1)
        # plt.imshow(plt_transformer(images[i]))
        #
        # plt.subplot(2, 4, 2)
        # plt.imshow(plt_transformer(resized_cams[i]))
        #
        # plt.subplot(2, 4, 3)
        # plt.imshow(plt_transformer(masks[i]))
        #
        # plt.subplot(2, 4, 4)
        # plt.imshow(plt_transformer(masked_images[i]))

        # # plt.subplot(2, 4, 6)
        # # plt.imshow(masked_image)
        #
        # plt.subplot(2, 4, 7)
        # plt.imshow(maskedBlur_image)
        #
        # plt.subplot(2, 4, 8)
        # plt.imshow(result_img)

        # plt.imshow(Image.fromarray(mask_02))
        #
        # plt.subplot(2, 4, 8)
        # plt.imshow(masked_image_02)
        # plt.title('masked_image_02')

        # plt.show()


        # break






if __name__ == '__main__':
    print('start')
    # func_1()

    # 创建全红图片
    batch_size = 4
    color_tensors = torch.zeros((batch_size, 3, 224, 224))
    color_tensors[:, 0, :, :] = 1.0    # red

    # # 查看图片
    # red_plt_image = plt_transformer(color_tensors[0])
    # red_plt_image.show()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    red_logits = ds_model(color_tensors).to(DEVICE)
    color_tensors.to(DEVICE)
    red_preds = torch.argmax(red_logits, 1)
    print(f'color preds: {red_preds}, \nlogits: {red_logits}, probs: {F.softmax(red_logits, 1)}')



























