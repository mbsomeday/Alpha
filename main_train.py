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

eff_model_obj = 'models.EfficientNet.efficientNetB0'
ped_model = train_ped_model_alpha(model_obj=eff_model_obj,
                 ds_name_list=['D2'],
                 batch_size=64,
                 reload=None,
                 epochs=150,
                 base_lr=0.01,
                 warmup_epochs=3,
                 lr_patience=5,
                 camLoss_coefficient=0.2,
                 ds_model_obj=eff_model_obj,
                 save_best_cls=False)
ped_model.train_model()





