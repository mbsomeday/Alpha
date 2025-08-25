import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


from data.dataset import my_dataset, dataset_from_dir
from utils.utils import load_model, get_obj_from_str, DEVICE


def test_ds_classifier(model_obj, ds_name_list, path_key, weights_path, batch_size, txt_name, opts=None):
    '''
        测试数据集分类模型
    :param model_obj:
    :param weights_path:
    :param batch_size:
    :return:
    '''
    ds_model = get_obj_from_str(model_obj)(num_class=3)
    ds_model = load_model(ds_model, weights_path).to(DEVICE)
    ds_model.eval()

    # ds_dataset = my_dataset(ds_name_list=ds_name_list, path_key=path_key, txt_name=txt_name)
    ds_dataset = dataset_from_dir(opts.dir_path, opts.label)
    ds_loader = DataLoader(ds_dataset, batch_size=batch_size, shuffle=False)
    print(len(ds_dataset))

    correct_num = 0
    y_pred = []
    y_label = []

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(ds_loader)):
            images = data_dict['image'].to(DEVICE)
            # ds_label = data_dict['ds_label'].to(DEVICE)
            ds_label = data_dict['label'].to(DEVICE)

            logits = ds_model(images)
            probs = torch.argmax(logits, 1)

            y_label.extend(ds_label.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

            correct_num += (ds_label == probs).sum()


        # 分别计算0，1，2的数量
        pred_as_D1 = y_pred.count(0)
        pred_as_D2 = y_pred.count(1)
        pred_as_D3 = y_pred.count(2)

        print(f'预测为来自D1:{pred_as_D1}, D2:{pred_as_D2}, D3:{pred_as_D3}')

        # cm = confusion_matrix(y_label, y_pred)
        # print(f'Testing cm:\n {cm}')
        #
        # ds_accuracy = correct_num / len(ds_dataset)
        # print(f'准确率为：{ds_accuracy}')
        #
        # bc = balanced_accuracy_score(y_label, y_pred)
        # print(f'balanced accuracy为：{bc}')





if __name__ == '__main__':
    print('Start')
    # model_obj = 'models.EfficientNet.efficientNetB0'
    # weights_path = r'D:\my_phd\Model_Weights\Stage5\EfficientNetB0_Scratch\efficientNetB0_dsCls-10-0.97636.pth'
    #
    # tt = test_ds_classifier(model_obj=model_obj,
    #                         weights_path=weights_path,
    #                         batch_size=8, txt_name='test.txt', path_key='Stage6_org', ds_name_list=['D4'])

    get_dtype = torch.get_default_dtype()
    print(get_dtype)
















