'''
By Xiangnan Zhang, 2025
School of Future Technologies, Beijing Institute of Technology.
Version for the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition
'''

import torch
torch.set_float32_matmul_precision('high')

from . import EEG
from . import training_utils
from . import models
