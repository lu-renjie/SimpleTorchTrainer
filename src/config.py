"""
This file records some configs that will not changed,
 and not affect the training results, eg. dataset path.
"""

import torch

device = torch.device('cuda')  # only support cuda, "cpu" or other device is not tested


CIFAR10_PATH = '/home/lurenjie/documents/datasets'
IMAGENET_PATH = '/mnt/data0/datasets/imagenet2012/'


