# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

# nms package does not work on Windows.
# so, we do not import packages that have nms dependency
if os.name != 'nt':
  from .mpii import MPIIDataset as mpii
  from .coco import COCODataset as coco
  from .coco_custom import COCOCustomDataset as coco_custom

from .coco_simple import COCOSimpleDataset as coco_simple
