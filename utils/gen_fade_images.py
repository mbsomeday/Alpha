'''
    生成fade images用于训练
'''
import argparse
from models.EfficientNet import efficientNetB0
from torch.utils.data import DataLoader
import torch, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torchcam.methods.gradient import GradCAM, LayerCAM
from torchvision import transforms
from PIL import Image

from utils import load_model, TemporaryGrad, save_image_tensor

from data.dataset import my_dataset


class gen_fade_images():
    def __init__(self, args):
        # 模型
        # ds_weights_path = R'D:\my_phd\Model_Weights\Stage5\EfficientNetB0_Scratch\efficientNetB0_dsCls-10-0.97636.pth'
        ds_model = efficientNetB0(num_class=3)
        self.ds_model = load_model(ds_model, args.ds_weights_path)
        self.ds_model.eval()

        self.feed_forward_features = None
        self.backward_features = None
        self.grad_layer = []

        # for i in range(9):
        #     cur_layer = f'features.{i}'
        #     self.grad_layer.append(cur_layer)

        self.grad_layer = ['features.7', 'features.8']
        # self.sigma = 0.25
        # self.omega = 100

        # 用torchcam
        self.cam_operator = LayerCAM(self.ds_model, target_layer=self.grad_layer)
        # for name, m in self.ds_model.named_modules():
        #     print(f'---{name}---')

        # 数据
        self.get_dataset = my_dataset(ds_name_list=args.ds_name_list, path_key='Stage6_org', txt_name=args.txt_name)
        self.get_loader = DataLoader(self.get_dataset, batch_size=1, shuffle=True)

        # self._register_hooks(self.ds_model, self.grad_layer)

        # 变量设置
        self.img_save_dir = args.img_save_dir


    def _register_hooks(self, model, grad_layer):
        '''
            为 ds_model 注册钩子函数
        '''
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Register forward hook and backward hook! Hooked layer: {self.grad_layer}")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)


    def calc_cam(self):
        for idx, data_dict in enumerate(tqdm(self.get_loader)):
            image = data_dict['image']
            img_name = data_dict['img_name'][0]
            img_path = data_dict['img_path']
            img_path[0].replace('\\', os.sep)
            contents = img_path[0].split(os.sep)
            cls_name = contents[-2]

            '''
                torchcam 生成 fade image
            '''
            logits = self.ds_model(image)
            pred = torch.argmax(logits, dim=1)
            cams = self.cam_operator(class_idx=pred.tolist(), scores=logits)

            trans_plt = transforms.ToPILImage()
            tensor_transform = transforms.ToTensor()

            resized_cam = []
            for idx, cam in enumerate(cams):
                cam = trans_plt(cam)
                cam = cam.resize((224, 224), resample=Image.Resampling.BICUBIC)
                cam = tensor_transform(cam)
                resized_cam.append(cam)

            # vis_cam = torch.cat(resized_cam, dim=0)
            # vis_cam = torch.sum(vis_cam, 0).unsqueeze(0)
            # (cam_min, cam_max) = (vis_cam.min(), vis_cam.max())
            # norm_cam = (vis_cam - cam_min) / (((cam_max - cam_min) + 1e-08)).data

            added_cam = torch.zeros([1, 224, 224])  # 初始化一个全0的tensor
            for cam in resized_cam:
                added_cam = torch.add(added_cam, cam)  # 逐个累加

            # 正则化
            (cam_min, cam_max) = (added_cam.min(), added_cam.max())
            added_cam = (added_cam - cam_min) / (((cam_max - cam_min) + 1e-08))

            print(f'added_cam:{added_cam.shape}， {added_cam.max()}')

            fade_image = image - added_cam * image

            fade_img_save_path = os.path.join(self.img_save_dir, cls_name, img_name)
            # save_image_tensor(fade_image, fade_img_save_path)

            # m, n = 3, 4
            # plt.figure(figsize=(16, 12))
            # cam_num = len(cams)
            # for i in range(1, cam_num+1):
            #     plt.subplot(m, n, i)
            #     plt.imshow(trans_plt(cams[i-1]))
            #     plt.title(f'feature.{i}')
            #
            # plt.subplot(m, n, cam_num+1)
            # plt.imshow(trans_plt(added_cam))
            # plt.title('added')
            #
            # plt.subplot(m, n, cam_num+2)
            # plt.imshow(trans_plt(image[0]))
            # plt.title('org')
            #
            # plt.subplot(m, n, cam_num+3)
            # plt.imshow(trans_plt(fade_image[0]))
            # plt.title('fade')
            #
            # plt.show()

            # 对原始图片减去 mask


            break


            '''
                代码生成 fade image
            '''
            # with TemporaryGrad():
            #     logits = self.ds_model(image)
            #     pred = torch.argmax(logits, dim=1)
            #     self.ds_model.zero_grad()
            #     grad_yc = logits[0, pred]
            #     grad_yc.backward(retain_graph=True)
            #     self.ds_model.zero_grad()
            #
            #     w = F.adaptive_avg_pool2d(self.backward_features, 1)  # shape: (batch_size, 1280, 1, 1)
            #     temp_w = w[0].unsqueeze(0)
            #     temp_fl = self.feed_forward_features[0].unsqueeze(0)
            #     ac = F.conv2d(temp_fl, temp_w)
            #     ac = F.relu(ac)
            #
            #     Ac = F.interpolate(ac, (224, 224))
            #
            #     # 获取mask
            #     Ac_max = Ac.max()
            #     Ac_min = Ac.min()
            #
            #     mask = Ac.detach().clone()
            #     mask.requires_grad = False
            #     mask = (mask - Ac_min) / (Ac_max - Ac_min)
            #     mask_max = mask.max()
            #     mask[mask < mask_max] = 0
            #     mask = mask * 0.5
            #     masked_image = image - image * mask
            #     # print(f'fade_img_save_path:{fade_img_save_path}, max:{mask_max}->{mask.max()}')
            #     save_image_tensor(masked_image, fade_img_save_path)
            #
            #     # from torchvision import transforms
            #     # plt_transform = transforms.ToPILImage()
            #     # plt.figure()
            #     # plt.subplot(221)
            #     # plt.imshow(plt_transform(image[0]))
            #     # plt.subplot(222)
            #     # plt.imshow(plt_transform(mask[0]))
            #     # plt.subplot(223)
            #     # plt.imshow(plt_transform(masked_image[0]))
            #     # plt.show()

            # if idx == 10:
            #     break
            # break

if __name__ == '__main__':
    torch.manual_seed(22)

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
        parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
        parser.add_argument('--txt_name', type=str, default='augmentation_train.txt')
        parser.add_argument('--img_save_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\layerCAM_aug_train')

        args = parser.parse_args()
        return args

    args = get_args()

    func = gen_fade_images(args)
    func.calc_cam()














