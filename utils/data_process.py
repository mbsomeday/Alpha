'''
    将分类好的行人和非行人划分为train/val/test
'''
import os, random, shutil
from tqdm import tqdm








def write_to_txt(sample_list, txt_path):
    '''
        将内容写到txt文件中
    '''
    with open(txt_path, 'a') as f:
        for item in sample_list:
            f.write(item)
    print(f'成功写入{len(sample_list)}个样本到{txt_path}!')


def split_smaples(txt_dir, num, sample_list):
    '''
        将samples按比例划分为train/val/test
    '''
    set_dir = {
        'train.txt': 0.6,
        'val.txt': 0.2,
        'test.txt': 0.2
    }

    for idx, set_name in enumerate(set_dir.keys()):
        set_num = int(set_dir[set_name] * num)
        txt_path = os.path.join(txt_dir, set_name)
        save_sample = sample_list[: set_num]
        sample_list = sample_list[set_num: ]
        write_to_txt(save_sample, txt_path)


def split_dataset(image_dir, num=None):
    '''

    :param base_dir:
    :param num:
    :return:
    '''
    images = os.listdir(image_dir)
    if num is None:
        num = len(images)

    print(f'total num:{num}')

    cls = image_dir.split(os.sep)[-1]
    print(cls)
    label = '1' if cls == 'pedestrian' else '0'
    info_list = []
    for img_name in images:
        contents = [os.path.join(cls, img_name), label]
        msg = ' '.join(contents) + '\n'
        info_list.append(msg)

    txt_dir = image_dir.replace(cls, 'dataset_txt')
    split_smaples(txt_dir, num, sample_list=info_list)

    # for cls_name in cls_list:
    #     cls_dir = os.path.join(base_dir, cls_name)
    #     sample_list = [os.path.join(cls_name, img_name) + 'pedestrian 1' if cls_name == 'pedestrian' else os.path.join(cls_name, img_name) + 'nonPedestrian 0' for img_name in os.listdir(cls_dir)]
    #     random.shuffle(sample_list)
    #     split_smaples(base_dir, num, sample_list)



def gather_images(image_list, dest_dir, num):
    '''
        如果获取的ped/noPed过多，则随机选取num个
    '''

    for idx in tqdm(range(num)):
        image_path = image_list[idx]
        image_contenst = image_path.split('\\')
        cp_to_path = os.path.join(dest_dir, image_contenst[-1])
        shutil.copy(image_path, cp_to_path)

'''
    根据bbox的IoU进行filter
'''
def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框之间的IoU(Intersection over Union)

    参数:
        bbox1: [x1, y1, x2, y2] 左上和右下坐标
        bbox2: [x1, y1, x2, y2] 左上和右下坐标

    返回:
        iou值 (float)
    """
    # 确定相交区域的坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # 如果没有相交区域，则IoU为0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算相交区域面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框各自的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集面积
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def filter_bboxes_by_iou(bbox_list, iou_threshold=0.5):
    """
    过滤边界框，返回与所有其他边界框IoU都小于阈值的边界框

    参数:
        bbox_list: 边界框列表，每个边界框格式为[x1, y1, x2, y2]
        iou_threshold: IoU阈值，默认为0.5

    返回:
        过滤后的边界框列表
    """
    filtered_bboxes = []
    n = len(bbox_list)

    for i in range(n):
        current_bbox = bbox_list[i]
        meets_criteria = True

        for j in range(n):
            if i == j:
                continue  # 不与自己比较

            iou = calculate_iou(current_bbox, bbox_list[j])
            if iou >= iou_threshold:
                meets_criteria = False
                break

        if meets_criteria:
            filtered_bboxes.append(current_bbox)

    return filtered_bboxes



def write_aug_train(base_dir):
    '''
        将目标数据集的aug写入augmentation_train.txt中
    '''
    info_list = []
    aug_dir = os.path.join(base_dir, 'augmentation_train')
    cls_names = os.listdir(aug_dir)

    for cls in cls_names:
        cls_dir = os.path.join(aug_dir, cls)
        image_list = os.listdir(cls_dir)
        cls_suffix = ' 0\n' if cls == 'nonPedestrian' else ' 1\n'
        for img_name in image_list:
            img_path = os.path.join('augmentation_train', cls, img_name)
            msg = img_path + cls_suffix
            info_list.append(msg)
    aug_train_path = os.path.join(base_dir, 'dataset_txt', 'augmentation_train.txt')
    print(f'向{aug_train_path}写入{len(info_list)}条数据。')
    with open(aug_train_path, 'a') as f:
        for item in info_list:
            f.write(item)



if __name__ == '__main__':
    random.seed(13)

    '''
        划分train/val/test 
    '''
    # image_dir = r'D:\my_phd\dataset\Stage6\stage6_ecp\pedestrian'
    # split_dataset(image_dir)

    '''
        写入augmentation train
    '''
    base_dir = r'D:\my_phd\dataset\Stage6\stage6_bdd100k'
    write_aug_train(base_dir)

















