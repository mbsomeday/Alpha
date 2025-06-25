'''
    本文件目的是对BDD100K数据集进行裁剪
    BDD100K的json文件非常大(1G)，所以用ijson来读取
'''
import ijson, os, random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


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

def crop_people(image_name, crop_box, image_dir, crop_image_name):

    x1, y1, x2, y2 = crop_box
    x1, x2 = calc_crop(x1, x2, max_val=image_width)
    y1, y2 = calc_crop(y1, y2, max_val=image_height)
    # print(x1, x2, y1, y2)
    crop_box = (x1, y1, x2, y2)
    # print(f'w:{x2-x1}, h:{y2-y1}')
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert('RGB')
    crop = image.crop(crop_box)
    crop_save_path = os.path.join(ped_crop_save_dir, crop_image_name)
    crop.save(crop_save_path)

    # 展示裁剪图片的代码
    # draw = ImageDraw.Draw(image)  # 创建绘图对象
    # draw.rectangle(crop_box, outline="red", width=2)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(crop)
    # plt.show()


def crop_image(name, label, min_size, max_size, image_dir):
    category = label.get('category', '')
    attributes = label.get('attributes', '')
    if category == 'person' and not attributes['occluded']:
        bbox = label.get('box2d', {})
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        crop_bbox = (x1, y1, x2, y2)
        w = x2 - x1
        h = y2 - y1
        if w > min_size and h > min_size and w < max_size and h < max_size:
            name_contents = name.split('.')
            crop_image_name = name_contents[0] + '_' + str(label['id']) + '.' + name_contents[1]    # 有label_id可以保证name不重复
            crop_people(name, crop_box=crop_bbox, image_dir=image_dir, crop_image_name=crop_image_name)

            return True
        else:
            return False
    else:
        return False


def read_json_and_image(json_list, image_dir_list):
    '''
        根据json中的annotation对原始图片进行裁剪
    '''
    no_people = 0
    for idx, json_file in enumerate(json_list):
        org_image_dir = image_dir_list[idx]
        print(idx, json_file, org_image_dir)

        no_people_list = []
        with open(json_file, 'r', encoding='utf-8') as f:
            items = ijson.items(f, 'item')
            for item in items:
                name = item.get('name', '')
                labels = item.get('labels', [])
                no_people_flag = True

                for label in labels:
                    category = label.get('category', '')
                    if category in ['rider', 'person', 'motor', 'bike']:
                        no_people_flag = False
                        break
                if no_people_flag:
                    no_people += 1
                    no_person_path = os.path.join(org_image_dir, name)
                    no_people_list.append(no_person_path)
                    # stop_flag = crop_image(name, label, min_size=min_size, max_size=max_size, image_dir=org_image_dir)
                    # if stop_flag:
                    #     pedestrian_num += 1

        with open(no_people_txt, 'a') as f:
            for no_p in no_people_list:
                f.write(no_p+'\n')

        print(f'不包含人的图片数量：{no_people}')


def crop_no_people(no_people_txt):
    '''
        从不包含people的图片中进行裁剪
    '''
    with open(no_people_txt, 'r') as f:
        data = f.readlines()
    # 只取5000个，所以打乱顺序
    random.shuffle(data)
    for idx, item in enumerate(tqdm(data)):
        item = item.strip()
        item_contents = item.split('\\')
        image = Image.open(item).convert('RGB')

        x1 = random.randint(0, 1055)
        y1 = random.randint(200, 335)

        x2 = x1 + 224
        y2 = y1 + 224

        crop_box = (x1, y1, x2, y2)
        crop = image.crop(crop_box)
        crop_image_path = os.path.join(nonPed_crop_save_dir, item_contents[-1])

        crop.save(crop_image_path)

        if idx == 5000:
            break




if __name__ == '__main__':
    # ----------------- seetings -----------------
    val_file = r'D:\my_phd\dataset\D4_BDD100K\bdd100k\labels\bdd100k_labels_images_val.json'
    train_file = r'D:\my_phd\dataset\D4_BDD100K\bdd100k\labels\bdd100k_labels_images_train.json'
    json_list = [val_file, train_file]

    val_image_dir = r'D:\my_phd\dataset\D4_BDD100K\bdd100k\images\100k\val'
    train_image_dir = r'D:\my_phd\dataset\D4_BDD100K\bdd100k\images\100k\train'
    image_dir_list = [val_image_dir, train_image_dir]  # 和json的顺序对应

    # 包含people部分的size设置
    min_size = 50
    max_size = 224

    image_width = 1280
    image_height = 720

    ped_crop_save_dir = r'D:\my_phd\dataset\D4_BDD100K\stage6_bdd100k\pedestrian'
    nonPed_crop_save_dir = r'D:\my_phd\dataset\D4_BDD100K\stage6_bdd100k\nonPedestrian'
    # print(f'min:{min_size}, max:{max_size}')

    no_people_txt = r'D:\my_phd\dataset\D4_BDD100K\stage6_bdd100k\noPeople.txt'
    # read_json_and_image(json_list, image_dir_list)

    crop_no_people(no_people_txt)






















