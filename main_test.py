import argparse

from training.training_template import DS_Classifier


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3'])
    # parser.add_argument('--txt_name', type=str, default='test.txt')
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--ds_model_obj', type=str, default='models.EfficientNet.efficientNetB0')

    parser.add_argument('--isTrain', action='store_true')

    # test
    parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--test_txt_name', type=str, default='test.txt')

    opts = parser.parse_args()
    return opts


opts = get_opts()

ds_classifier = DS_Classifier(opts)
ds_classifier.test()











