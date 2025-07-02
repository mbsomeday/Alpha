# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from training.training_template import Ped_Classifier
from configs.pedCls_args import BaseArgs, TrainArgs, TestArgs


# opts = TestArgs().parse()
# ped_cls = Ped_Classifier(opts=opts)
# ped_cls.test()


# from test_func.ds_cls import test_ds_classifier
def get_opts():
    parser = argparse.ArgumentParser()
    """Define the common options that are used in both training and test."""
    parser.add_argument('--weights_path', type=str)
    return parser.parse_args()
#
# opts = get_opts()
# weights_path = opts.weights_path
# ds_cls = test_ds_classifier(model_obj='models.EfficientNet.efficientNetB0',
#                             ds_name_list=['D1', 'D2'],
#                             batch_size=96,
#                             weights_path=weights_path,
#                             txt_name='test.txt',
#                             path_key='Stage6_org'
#                             )


from test_func.ds_cls import test_ds_classifier


test_ds_classifier(model_obj='models.EfficientNet.efficientNetB0',
                   ds_name_list=['D1','D4'],
                   path_key='Stage6_org',
                   weights_path=r'/kaggle/input/stage5-weights-effidscls/efficientNetB0_dsCls-10-0.97636.pth',
                   batch_size=96,
                   txt_name='val.txt'
                   )





