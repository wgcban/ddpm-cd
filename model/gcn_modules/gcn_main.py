# The decoder in our RSCD model
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skimage.segmentation import slic