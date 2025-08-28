# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py

import argparse


class BaseArgs():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False


    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--ds_model_obj', type=str, default='models.EfficientNet.efficientNetB0')
        parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3'], help='dataset list')
        parser.add_argument('--ds_labels', nargs='+', default=[0, 1, 2])

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--data_key', type=str, default='Stage6_org')
        parser.add_argument('--isTrain', action='store_true')
        parser.add_argument('--rand_seed', type=int, default=3)

        self.initialized = True
        return parser


    def parser(self):
        parser = argparse.ArgumentParser()
        if not self.initialized:
            parser = self.initialize(parser)
        # get basic args
        opt, _ = parser.parse_known_args()

        return opt


class TrainArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = BaseArgs.initialize(self, parser)

        # model
        parser.add_argument('--init_method', type=str, default='orthogonal', help='the way to initialize model weights, e.g., kaiming, orthogonal')
        parser.add_argument('--base_lr', type=float, default=0.01)

        return parser

class TestArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = BaseArgs.initialize(self, parser)

        parser.add_argument('--test_txt_name', type=str, default='test.txt')
        parser.add_argument('--test_batch_size', type=int, default=32)
        parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
        # parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\reMapDatasetClassifierLabel\dsClsD1D2D3_210-40-0.32013.pth')

        return parser















