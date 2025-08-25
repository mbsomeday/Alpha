# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse
# from training.training_template import ds_classifier


'''
训练ped classifier
'''
# from training.training_template import Ped_Classifier
# from configs.pedCls_args import TrainArgs
#
# opts = TrainArgs().parse()
# ped_cls = Ped_Classifier(opts=opts)
# ped_cls.train()


'''
    训ds classifier
'''

from training.training_template import DS_Classifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_model_obj', type=str, default='models.EfficientNet.efficientNetB0')
    parser.add_argument('--isTrain', action='store_true')

    parser.add_argument('--ds_labels', nargs='+', default=[2, 1, 0])
    parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3'], help='dataset list')
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)

    parser.add_argument('--base_lr', type=float, default=0.01)

    # callbacks
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--max_epochs', type=int, default=150, help='max epochs for training')
    parser.add_argument('--warmup_epochs', type=int, default=3)

    # test
    parser.add_argument('--ds_weights_path', type=str)
    parser.add_argument('--test_txt_name', type=str)


    args = parser.parse_args()

    return args


opts = get_args()

my_ds_classifier = DS_Classifier(opts)
# my_ds_classifier.train()

my_ds_classifier.test()



'''
    旧代码训练ds classifier
'''
# from training.training import train_ds_model_alpha
# model_obj = 'models.EfficientNet.efficientNetB0'
# ds_cls = train_ds_model_alpha(model_obj=model_obj,
#                               batch_size=64,
#                               path_key='Stage6_org',
#                               ds_name_list=['D1', 'D2'],
#                               warmup_epochs=3,
#                               lr_patience=5
#                               )
# ds_cls.train_model()

'''
    旧代码训练ped classifier
'''
# from training.training import train_ped_model_alpha
#
#
# def initialize():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--ds_name_list', nargs='+', default=['D1'], help='dataset list')
#     parser.add_argument('--ds_weights', type=str, default=None)
#     parser.add_argument('--key_path', type=str)
#     parser.add_argument('--camLoss_coefficient', type=float)
#
#     args = parser.parse_args()
#
#     return args
#
# eff_model_obj = 'models.EfficientNet.efficientNetB0'
# args = initialize()
# ds_weights = args.ds_weights
# ds_name_list = args.ds_name_list
# ds_key_path = args.key_path
# camLoss_coefficient = args.camLoss_coefficient if args.camLoss_coefficient > 0.0 else None
#
#
# ped_model = train_ped_model_alpha(model_obj=eff_model_obj,
#                  ds_name_list=ds_name_list,
#                  ds_key_path=ds_key_path,
#                  batch_size=64,
#                  reload=None,
#                  epochs=150,
#                  base_lr=0.01,
#                  warmup_epochs=3,
#                  lr_patience=5,
#                  camLoss_coefficient=camLoss_coefficient,
#                  ds_model_obj=eff_model_obj,
#                  ds_weights=ds_weights,
#                  save_best_cls=False)
#
# ped_model.train_model()





