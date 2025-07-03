# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

# from training.training_template import Ped_Classifier
# from configs.pedCls_args import TrainArgs
#
# opts = TrainArgs().parse()
# ped_cls = Ped_Classifier(opts=opts)
# ped_cls.train()


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


from training.training import train_ped_model_alpha


def initialize():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name_list', nargs='+', default=['D1'], help='dataset list')
    parser.add_argument('--ds_weights', type=str, default=None)
    parser.add_argument('--key_path', type=str)
    parser.add_argument('--camLoss_coefficient', type=float)

    args = parser.parse_args()

    return args

eff_model_obj = 'models.EfficientNet.efficientNetB0'
args = initialize()
ds_weights = args.ds_weights
ds_name_list = args.ds_name_list
ds_key_path = args.key_path
camLoss_coefficient = args.camLoss_coefficient if args.camLoss_coefficient > 0.0 else None


ped_model = train_ped_model_alpha(model_obj=eff_model_obj,
                 ds_name_list=ds_name_list,
                 ds_key_path=ds_key_path,
                 batch_size=64,
                 reload=None,
                 epochs=150,
                 base_lr=0.01,
                 warmup_epochs=3,
                 lr_patience=5,
                 camLoss_coefficient=camLoss_coefficient,
                 ds_model_obj=eff_model_obj,
                 ds_weights=ds_weights,
                 save_best_cls=False)

ped_model.train_model()





