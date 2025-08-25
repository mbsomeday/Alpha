import argparse

# from training.training_template import DS_Classifier
from test_func.ds_cls import test_ds_classifier


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3'])
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--ds_model_obj', type=str, default='models.EfficientNet.efficientNetB0')

    parser.add_argument('--isTrain', action='store_true')

    # test
    parser.add_argument('--weights_path', type=str, default=r'D:\chrom_download\dsClsD1D2D3-03-16.15992.pth')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--test_txt_name', type=str, default='test.txt')

    # dataset
    parser.add_argument('--dir_path', type=str, default=r'D:\my_phd\dataset\Stage6\black_masked\blackMask_D2')
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)

    opts = parser.parse_args()
    return opts


opts = get_opts()

# ds_classifier = DS_Classifier(opts)
# ds_classifier.test()

model_obj = opts.ds_model_obj
ds_name_list = opts.ds_name_list
path_key = opts.path_key
weights_path = opts.weights_path
batch_size = opts.batch_size
txt_name = 'test.txt'
test_ds_classifier(model_obj, ds_name_list, path_key, weights_path, batch_size, txt_name, opts=opts)








