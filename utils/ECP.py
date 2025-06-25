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
    # noPed_num = 0
    # no_ped_list = []
    for json_path in tqdm(json_list):
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
        # if len(objectives) == 0:
        #     noPed_num += 1
        #     image_path = json_path.replace('labels', 'img')
        #     image_path = image_path.replace('json', 'png')
        #     no_ped_list.append(image_path + '\n')
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
        # ped_num += len(bbox_list)

        for crop_idx, crop_box in enumerate(bbox_list):
            crop_image_name = path_contents[-1].split('.')[0] + '_' + str(crop_idx) + '.jpg'
            crop_pedestrian(image_path, crop_box, crop_image_name)


                # stop_flag = True
                # cropped_num += 1
            # if cropped_num == 4000:
            #     stop_flag = True

        # if stop_flag:
        #     break

    # with open(no_ped_path, 'a') as f:
    #     for item in no_ped_list:
    #         f.write(item)

    # print(f'符合条件的行人：{ped_num}')
    # print(f'没有行人的：{noPed_num}')

def crop_night_no_people(no_people_txt):
    '''
        从night图片中裁剪no people，由于数量有限，每张图片裁剪5个，选取的图片中不包含任何标注
    '''

    def save_crop(item, image, x1, y1, obj_idx=None):
        x2 = x1 + 224
        y2 = y1 + 224
        crop_box = (x1, y1, x2, y2)
        crop = image.crop(crop_box)
        crop_name = item.split('\\')[-1]
        if obj_idx is not None:
            crop_name = crop_name.replace('.', '_' + obj_idx + '.')
        crop_image_path = os.path.join(no_ped_crop_dir, crop_name)
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
        # for i in range(5):
        #     x_start = i * 384
        #     x1 = random.randint(x_start, x_start + 160)
        #     y1 = random.randint(360, 596)
        #     x2 = x1 + 224
        #     y2 = y1 + 224
        #     crop_box = (x1, y1, x2, y2)
        #     crop = image.crop(crop_box)
        #     crop_name = item.split('\\')[-1]
        #     crop_name = crop_name.replace('.', '_'+str(i)+'.')
        #     crop_image_path = os.path.join(no_ped_crop_dir, crop_name)
        #     crop.save(crop_image_path)

        '''
            裁剪day非行人
        '''
        two_crops_prob = random.random()
        if two_crops_prob > 0.8:        # 从该图片中裁剪2个crop
            for i in range(2):
                x_start = 300 if i == 0 else 650
                x_end = (650 - 244) if i == 0 else (1600 - 244)
                x1 = random.randint(x_start, x_end)
                y1 = random.randint(360, 596)
                save_crop(item, image, x1, y1, obj_idx=str(i))

        else:       # 从该图片中裁剪2个crop
            x1 = random.randint(300, 1600 - 224)
            y1 = random.randint(360, 596)
            save_crop(item, image, x1, y1)

        # break




if __name__ == '__main__':
    json_dir = r'E:\Dataset\ECP\day\labels\train'
    image_dir = r'E:\Dataset\ECP\day\img\train'
    ped_crop_save_dir = r'D:\my_phd\dataset\Stage6\stage6_ecp\ped_day'
    # no_people_txt = r'D:\my_phd\dataset\Stage6\stage6_ecp\no_ped_day.txt'
    # no_ped_crop_dir = r'D:\my_phd\dataset\Stage6\stage6_ecp\noPed_day'
    #
    min_size = 50
    max_size = 224
    image_width = 1920
    image_height = 1024
    json_list = []

    location_list = os.listdir(json_dir)

    for dir_name in tqdm(location_list):
        dir_path = os.path.join(json_dir, dir_name)
        json_list.extend(os.path.join(dir_path, j_name) for j_name in os.listdir(dir_path))

    # random.shuffle(json_list)
    print(f'num of json: {len(json_list)}')
    read_json(json_list)

    # crop_night_no_people(no_people_txt)
























