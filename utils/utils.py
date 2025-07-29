import torch, importlib, os, re
from sklearn.metrics import confusion_matrix
import pandas as pd
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from models.VGG import vgg16_bn
from configs.paths_dict import PATHS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_cm(y_true, y_pred, label_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(f'cm:\n {cm}')
    # conf_matrix_df = pd.DataFrame(cm, columns=label_names, index=label_names)
    # sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    # plt.title(title)
    # plt.ylabel('Label')
    # plt.xlabel('Prediction')
    # # plt.savefig(f'{title}.png')
    # print(f'Image save to {title}.png')
    # # plt.show()



def get_vgg_DSmodel():
    '''
        获取 VGG dataset classifier
    '''
    model = vgg16_bn(num_class=4)
    weight_path = PATHS['ds_cls_ckpt']
    print(f'Loading dataset classifier: {weight_path}')
    checkpoints = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoints['model_state_dict'])
    return model


def get_orgPed_model(ds_name):
    model = vgg16_bn(num_class=2)
    weight_path = PATHS['ped_cls_ckpt'][ds_name]
    print(f'Loading model: {weight_path}')
    checkpoints = torch.load(weight_path, map_location=DEVICE, weights_only=True if DEVICE=='cuda' else False)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model


def load_model(model, weights_path):
    print(f'Loading model from {weights_path}')
    ckpts = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model.load_state_dict(ckpts['model_state_dict'])
    return model

class TemporaryGrad(object):
    '''
    https://blog.csdn.net/qq_44980390/article/details/123672147
    '''
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


def get_obj_from_str(in_str):
    '''
        根据 str类型的函数名 来调用函数
    '''
    module, cls = in_str.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class DotDict(dict):
    '''
        将字典转换为可直接用 . 调用的对象
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setitem__(self, key, value):
        '''
            保证 dict[key] 的形式可以更改值
        '''
        if isinstance(value, dict):
            value = DotDict(value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        '''
            保证 dict.key 的形式可以更改值
        '''
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value





def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)



def get_gpu_info():
    '''
        打印出显卡信息
    '''
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print(f'GPU num: {n_gpu}, g_name: {gpu_name}')



def train_val_graph(txt_path, title, save_path):
    '''
    从txt文件中读取信息并绘制graph
    '''
    with open(txt_path, 'r') as f:
        data = f.readlines()
    epoch_list = []
    train_ba = []
    train_loss = []
    val_ba = []
    val_loss = []
    for idx, item in enumerate(data):
        epoch_search = re.search(r'-+ Epoch: (\d+) -+', item)
        train_search = re.findall(r'Train: org_bc: (\d+\.\d+).*?operated_bc: (\d+\.\d+).*?accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)
        val_search = re.findall(r'Val: accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)

        if epoch_search is not None:
            epoch_list.append(int(epoch_search.group(1)))

        if train_search:
            org_bc, operated_bc, accuracy, train_ba_val, train_loss_val = train_search[0]
            train_ba.append(float(train_ba_val)*100/100)
            train_loss.append(float(train_loss_val)*100/100)

        if val_search:
            _, val_ba_val, val_loss_val = val_search[0]
            val_ba.append(float(val_ba_val)*100/100)
            val_loss.append(float(val_loss_val)*100/100)

    # print(f'train_ba:{len(train_ba)}')
    # print(f'train_loss:{len(train_loss)}')
    # print(f'val_ba:{len(val_ba)}')
    # print(f'val_loss:{len(val_loss)}')

    '''
    绘制两张图
    '''
    # # 绘制曲线
    # plt.figure(figsize=(12, 8))
    #
    # # 左侧Y轴：Loss
    # plt.subplot(2, 1, 1)  # 2行1列的上半部分
    # plt.plot(epoch_list, train_loss, 'r-', label='Train Loss', marker='o')
    # plt.plot(epoch_list, val_loss, 'b--', label='Val Loss', marker='s')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss', color='black')
    # plt.tick_params(axis='y', labelcolor='black')
    # plt.grid(linestyle='--', alpha=0.5)
    # plt.legend()
    #
    # # 右侧Y轴：Accuracy/Balanced Accuracy
    # plt.subplot(2, 1, 2)  # 2行1列的下半部分
    # plt.plot(epoch_list, train_ba, 'r-', label='Train BA', marker='^')
    # plt.plot(epoch_list, val_ba, 'b--', label='Val BA', marker='v')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy', color='black')
    # plt.tick_params(axis='y', labelcolor='black')
    # plt.grid(linestyle='--', alpha=0.5)
    # plt.legend()
    #
    # # 添加标题
    # plt.suptitle('Training and Validation Metrics (Model: efficientNetB0, Dataset: D1)', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    '''
    绘制一张图
    '''
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制两条Loss曲线（左侧Y轴）sd
    ax1.plot(epoch_list, train_loss, label='Train loss', color='red', linestyle='-', marker='o')
    ax1.plot(epoch_list, val_loss, label='Val loss', color='orange', linestyle='--', marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')  # 左侧Y轴标签颜色

    # 创建右侧Y轴（共享同一个X轴）
    ax2 = ax1.twinx()

    # 绘制两条Accuracy曲线（右侧Y轴）
    ax2.plot(epoch_list, train_ba, label='Train BA', color='blue', linestyle='-.', marker='^')
    ax2.plot(epoch_list, val_ba, label='Val BA', color='green', linestyle=':', marker='*')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')  # 右侧Y轴标签颜色

    # 合并图例（将左右Y轴的图例合并显示）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # 添加标题
    plt.title(title, fontsize=14, pad=20)

    # 调整布局防止图例遮挡
    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()



def plot_two_model(txt1, txt2):
    def get_bs_info(data):
        epoch_list = []
        train_ba = []
        train_loss = []
        val_ba = []
        val_loss = []
        for idx, item in enumerate(data):
            epoch_search = re.search(r'-+ Epoch: (\d+) -+', item)
            train_search = re.findall(r'Train: accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)
            val_search = re.findall(r'Val: accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)

            if epoch_search is not None:
                epoch_list.append(int(epoch_search.group(1)))

            if train_search:
                train_acc_val, train_ba_val, train_loss_val = train_search[0]
                train_ba.append(float(train_ba_val) * 100 / 100)
                train_loss.append(float(train_loss_val) * 100 / 100)

            if val_search:
                _, val_ba_val, val_loss_val = val_search[0]
                val_ba.append(float(val_ba_val) * 100 / 100)
                val_loss.append(float(val_loss_val) * 100 / 100)

        return epoch_list, train_ba, train_loss, val_ba, val_loss
    def get_fade_info(data):
        epoch_list = []
        train_ba = []
        train_loss = []
        val_ba = []
        val_loss = []
        for idx, item in enumerate(data):
            epoch_search = re.search(r'-+ Epoch: (\d+) -+', item)
            train_search = re.findall(r'Train: org_bc: (\d+\.\d+).*?operated_bc: (\d+\.\d+).*?accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)
            val_search = re.findall(r'Val: accuracy: (\d+\.\d+).*?balanced_accuracy: (\d+\.\d+).*?loss: (\d+\.\d+)', item)

            if epoch_search is not None:
                epoch_list.append(int(epoch_search.group(1)))

            if train_search:
                org_bc, operated_bc, accuracy, train_ba_val, train_loss_val = train_search[0]
                train_ba.append(float(train_ba_val) * 100 / 100)
                train_loss.append(float(train_loss_val) * 100 / 100)

            if val_search:
                _, val_ba_val, val_loss_val = val_search[0]
                val_ba.append(float(val_ba_val) * 100 / 100)
                val_loss.append(float(val_loss_val) * 100 / 100)

        return epoch_list, train_ba, train_loss, val_ba, val_loss


    with open(txt1, 'r') as f:
        data1 = f.readlines()

    with open(txt2, 'r') as f:
        data2 = f.readlines()

    # 获取loss和train
    # epoch_list, train_ba, train_loss, val_ba, val_loss
    info1 = get_bs_info(data1)
    info2 = get_fade_info(data2)

    # 绘制train loss对比
    plt.figure(figsize=(8, 6))  # 可选的图形大小设置
    # plt.plot(info1[0], info1[1], label='Baseline Train BA', color='red', linestyle='-', marker='o') # 添加 label 用于图例，marker 显示数据点
    # plt.plot(info2[0], info2[1], label='08Pepper Train BA', color='orange', linestyle='--', marker='s')  # 使用不同的 marker 样式

    plt.plot(info1[0], info1[3], label='Baseline Val BA', color='red', linestyle='-', marker='o') # 添加 label 用于图例，marker 显示数据点
    plt.plot(info2[0], info2[3], label='08Pepper Val BA', color='orange', linestyle='--', marker='s')  # 使用不同的 marker 样式

    # 添加标题和标签
    title_metric = 'balanced accuracy'
    plt.title(f'Baseline VS 0.2Fade {title_metric}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

    # fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    # # 绘制两条Loss曲线（左侧Y轴）
    # ax1.plot(info1[0], info1[2], label='Baseline Train loss', color='red', linestyle='-', marker='o')
    # ax1.plot(info2[0], info2[2], label='0.2Fade Train loss', color='orange', linestyle='--', marker='s')
    # ax1.set_xlabel('Epoch', fontsize=12)
    # ax1.set_ylabel('Loss', fontsize=12)
    # ax1.tick_params(axis='y', labelcolor='red')  # 左侧Y轴标签颜色
    #
    # # 创建右侧Y轴（共享同一个X轴）
    # ax2 = ax1.twinx()
    #
    # # 绘制两条Accuracy曲线（右侧Y轴）
    # ax2.plot(info2[0], info2[2], label='zzz', color='blue', linestyle='-.', marker='^')
    # ax2.plot(info2[0], info2[4], label='ccc', color='green', linestyle=':', marker='*')
    # ax2.set_ylabel('Accuracy', fontsize=12)
    # ax2.tick_params(axis='y', labelcolor='blue')  # 右侧Y轴标签颜色
    #
    # # 合并图例（将左右Y轴的图例合并显示）
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    #
    # # 添加标题
    # plt.title('Baseline VS 0.2Fade', fontsize=14, pad=20)
    #
    # # 调整布局防止图例遮挡
    # plt.tight_layout()
    # # plt.savefig(save_path)
    # plt.show()

























if __name__ == '__main__':
    '''
    画一个model的graph
    '''
    # txt_path = r'D:\my_phd\Model_Weights\Stage6\new_dataset\efficientNetB0_D3_90_0.2fadeLoss\Train_info.txt'
    # set = 'AugTrain'
    # monitor = 'loss'
    # # monitor = 'balanced accuracy'
    # seed = 90
    # title = f'Train on D1 {set} set\nMonitoring {monitor} (seed:{seed})'
    # save_name = f'D1{set}-{monitor}-{seed}.png'
    # save_dir = r'D:\my_phd\Model_Weights\Stage6\对比loss和ba\figures'
    # save_path = os.path.join(save_dir, save_name)
    # train_val_graph(txt_path, title=title, save_path=save_path)

    '''
    画两个model的graph
    '''
    txt1 = r'D:\my_phd\Model_Weights\Stage6\new_dataset\efficientNetB0_D3_3_baseline\Train_info.txt'
    txt2 = r'D:\my_phd\Model_Weights\Stage6\new_dataset\trainOnlyFade\val_fade\efficientNetB0_D3_39_08CAPepper\Train_info.txt'
    plot_two_model(txt1, txt2)






















































