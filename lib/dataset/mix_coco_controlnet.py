# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import copy
from core.config import config
from dataset.JointsDataset import JointsDataset
from dataset.coco import COCODataset
from dataset.controlnet import ControlNetDataset
from utils.rand_augment import RandAugment
import logging
logger = logging.getLogger(__name__)

class Mixed_COCO_ControlNet_Dataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.coco = COCODataset(cfg, root, image_set, is_train, transform)
        # Set Length as Full
        cfg.DATASET.TRAIN_LEN = 0
        self.controlnet = ControlNetDataset(cfg, root, image_set, is_train, transform)

        np.random.seed(1314)

        self.controlnet_size = len(self.controlnet.db)
        self.coco_size = len(self.coco.db)

        self.cfg = cfg
        logger.info('=> Total load {} controlnet samples'.format(len(self.controlnet.db)))

    def __len__(self):

        return self.coco_size + self.controlnet_size

    def __getitem__(self, idx):

        if idx < self.coco_size:
            return self.coco.__getitem__(idx)
        else:
            return self.controlnet.__getitem__(idx - self.coco_size)
