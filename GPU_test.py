import os
import torch
'''import argparse
import numpy as np
import sys
sys.path.append('')
from scipy import misc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from pytorch_lib.utils.dataset import test_dataset as EvalDataset
from pytorch_lib.lib.DGNet import DGNet as Network
'''

import torch.backends.cudnn as cudnn
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))