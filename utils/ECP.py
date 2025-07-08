import os, json
import random
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data_process import filter_bboxes_by_iou


def calc_crop(a, b, max_val):
    diff = b - a
    # print(f'a, b:{a}, {b}, max:{max_val}, diff:{diff}')
    if diff == 224:
        return a, b
    elif diff < 224:
        change_size = 224 - diff
        if change_size == 1:
            if a != 0:
                a = max(0, (a - 1))
                return calc_crop(a, b, max_val)
            else:
                b = min((b + 1), max_val)
                return calc_crop(a, b, max_val)
        elif a == 0:
            b = min((b + change_size), max_val)
            return calc_crop(a, b, max_val)
        elif b == max_val:
            a = max(0, (a - change_size))
            return calc_crop(a, b, max_val)
        else:
            temp_val = int(change_size/2)
            a = max(0, (a - temp_val))
            b = min((b + change_size - temp_val), max_val)
            return calc_crop(a, b, max_val)
    else:
        raise ValueError(f'diff too large{diff}')

def crop_pedestrian(image_path, crop_box, crop_image_name):
    x1, y1, x2, y2 = crop_box
    x1, x2 = calc_crop(x1, x2, max_val=image_width)
    y1, y2 = calc_crop(y1, y2, max_val=image_height)
    crop_box = (x1, y1, x2, y2)

    image = Image.open(image_path).convert('RGB')
    crop = image.crop(crop_box)

    crop_save_path = os.path.join(ped_crop_save_dir, crop_image_name)
    crop.save(crop_save_path)

    # # 展示裁剪图片的代码
    # draw = ImageDraw.Draw(image)  # 创建绘图对象
    # draw.rectangle(crop_box, outline="red", width=2)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(crop)
    # plt.show()

def read_json(json_list):
    ped_num = 0
    # cropped_num = 0
    noPed_num = 0
    # no_ped_list = []
    # ped_identity = ['pedestrian', 'buggy-group', 'person-group-far-away', 'rider']

    for idx, json_path in enumerate(tqdm(json_list)):
        no_ped_flag = True
        # stop_flag = False
        bbox_list = []
        path_contents = json_path.split('\\')
        image_path = os.path.join(image_dir, path_contents[-2], path_contents[-1].split('.')[0]+'.png')
        with open(json_path, 'r') as f:
            data = json.load(f)
        objectives = data['children']
        '''
            获取无行人的图片
        '''
        # for obj in objectives:
            # idt = obj['identity']
            # if idt in ped_identity:
            #     no_ped_flag = False
            #     break
        # if no_ped_flag:
        #     image_path = json_path.replace('labels', 'img')
        #     image_path = image_path.replace('json', 'png')
            # no_ped_list.append(image_path + '\n')
            # noPed_num += 1

        # if len(objectives) == 0:
        #     noPed_num += 1
            # image_path = json_path.replace('labels', 'img')
            # image_path = image_path.replace('json', 'png')
            # no_ped_list.append(image_path + '\n')
        '''
            裁剪pedestrian crop
        '''
        for obj_idx, obj in enumerate(objectives):
            obj_cls = obj['identity']
            x0, y0, x1, y1 = int(obj['x0']), int(obj['y0']), int(obj['x1']), int(obj['y1'])
            w = x1 - x0
            h = y1 - y0
            crop_box = (x0, y0, x1, y1)
            tags = obj['tags']
            if obj_cls == 'pedestrian' and w > min_size and h > min_size and w < max_size and h < max_size and len(tags) == 0:
                bbox_list.append(crop_box)
        bbox_list = filter_bboxes_by_iou(bbox_list)
        ped_num += len(bbox_list)
        for crop_idx, crop_box in enumerate(bbox_list):
            crop_image_name = path_contents[-1].split('.')[0] + '_' + str(crop_idx) + '.jpg'
            crop_pedestrian(image_path, crop_box, crop_image_name)

        if ped_num >= 2500:
            break

    # print(f'行人数量:{ped_num}')
    # print(f'没有行人的图片：{noPed_num}')
    # with open(no_people_txt, 'a') as f:
    #     for item in no_ped_list:
    #         f.write(item)


def crop_no_people(no_people_txt):
    '''
        从night图片中裁剪no people，由于数量有限，每张图片裁剪n个，选取的图片中不包含任何标注
    '''

    def save_crop(item, image, x1, y1, obj_idx=None):
        x2 = x1 + 224
        y2 = y1 + 224
        crop_box = (x1, y1, x2, y2)
        crop = image.crop(crop_box)
        crop_name = item.split('\\')[-1]
        if obj_idx is not None:
            crop_name = crop_name.replace('.', '_' + obj_idx + '.')
        crop_name = crop_name.replace('png', 'jpg')
        crop_image_path = os.path.join(no_ped_crop_dir, crop_name)
        # print(f'crop_image_path:{crop_image_path}')
        crop.save(crop_image_path)

        # # 展示裁剪图片的代码
        # draw = ImageDraw.Draw(image)  # 创建绘图对象
        # draw.rectangle(crop_box, outline="red", width=2)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(crop)
        # plt.show()

    with open(no_people_txt, 'r') as f:
        data = f.readlines()
    for idx, item in enumerate(tqdm(data)):
        item = item.strip()
        image = Image.open(item).convert('RGB')

        '''
            裁剪night非行人
        '''
        # for i in range(3):
        #     x_start = i * 640
        #     x1 = random.randint(x_start, x_start + 416)
        #     y1 = random.randint(360, 596)
        #     save_crop(item, image, x1, y1, obj_idx=str(i))

        '''
            裁剪day非行人
        '''
        x1 = random.randint(300, 1600 - 224)
        y1 = random.randint(360, 596)
        save_crop(item, image, x1, y1)

        # break


        # if two_crops_prob > 0.8:        # 从该图片中裁剪2个crop
        #     for i in range(2):
        #         x_start = 300 if i == 0 else 650
        #         x_end = (650 - 244) if i == 0 else (1600 - 244)
        #         x1 = random.randint(x_start, x_end)
        #         y1 = random.randint(360, 596)
        #         save_crop(item, image, x1, y1, obj_idx=str(i))
        # else:       # 从该图片中裁剪2个crop
        #     x1 = random.randint(300, 1600 - 224)
        #     y1 = random.randint(360, 596)
        #     save_crop(item, image, x1, y1)

        # break


def new_crop_no_pedestrian(no_people_list):
    '''
        从不包含people的图片中进行裁剪
    '''
    with open(no_people_txt, 'r') as f:
        data = f.readlines()

    print(f'num of no people samples:{len(data)}')

    for idx, item in enumerate(tqdm(data)):
        item = item.strip()
        item_contents = item.split('\\')
        image = Image.open(item).convert('RGB')

        x1 = random.randint(0, 1376)
        y1 = random.randint(360, 596)

        x2 = x1 + 224
        y2 = y1 + 224

        crop_box = (x1, y1, x2, y2)
        crop = image.crop(crop_box)
        crop_image_path = os.path.join(no_ped_crop_dir, item_contents[-1])
        crop.save(crop_image_path)
        # print(crop_image_path)
        # break


if __name__ == '__main__':
    json_dir = r'E:\Dataset\ECP\day\labels\train'
    base_dir = r'D:\my_phd\dataset\Stage6\stage6_ecp'
    # cls_name = 'nonPedestrian'
    cls_name = 'pedestrian'

    # no_people_txt = os.path.join(base_dir, 'noPed_day_train.txt')
    image_dir = json_dir.replace('labels', 'img')
    # no_ped_crop_dir = os.path.join(base_dir, 'noPed_day_train')
    ped_crop_save_dir = os.path.join(base_dir, 'ped_day')
    # crop_save_dir = os.path.join(base_dir, cls_name)
    # if not os.path.exists(crop_save_dir):
    #     print(f'创建{crop_save_dir}')
    #     os.mkdir(crop_save_dir)

    min_size = 50
    max_size = 224
    image_width = 1920
    image_height = 1024
    json_list = []

    location_list = os.listdir(json_dir)
    for dir_name in tqdm(location_list):
        dir_path = os.path.join(json_dir, dir_name)
        json_list.extend(os.path.join(dir_path, j_name) for j_name in os.listdir(dir_path))
    random.shuffle(json_list)
    print(f'num of json in {json_dir}: {len(json_list)}')
    read_json(json_list)

    # crop_no_people(no_people_txt)


























