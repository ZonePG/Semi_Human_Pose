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

class Mixed_COCO_GC_ControlNet_Dataset(JointsDataset):

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.sup_coco = COCODataset(cfg, root, image_set, is_train, transform)
        self.controlnet = ControlNetDataset(cfg, root, image_set, is_train, transform)
        cfg.DATASET.TRAIN_LEN = 0
        self.unsup_coco = COCODataset(cfg, root, image_set, is_train, transform)

        np.random.seed(1314)

        self.sup_coco_size = len(self.sup_coco.db)
        self.controlnet_size = len(self.controlnet.db)
        self.unsup_coco_size = len(self.unsup_coco.db)
        self.sets = [self.sup_coco, self.controlnet, self.unsup_coco]
        self.sizes = [self.sup_coco_size, self.controlnet_size, self.unsup_coco_size]

        self.group_size = max(self.sizes)
        self.cfg = cfg
        self.shuffle_ind()

    def __len__(self):
        return self.group_size

    def shuffle_ind(self):
        self.data_ind = np.random.choice((self.sizes)[0], self.group_size)
        return None

    def __getitem__(self, idx):

        # Set the COCO joints as joints vis of AI
        ai_joint_ind = [] 
        mapping = self.sets[0].u2a_mapping
        for i in range(len(mapping)):
            if mapping[i] != '*':
                ai_joint_ind.append(int(i))

        # Pre define the rotation angle, scale and flip
        rf = self.sets[0].rotation_factor
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
            if random.random() <= 0.6 else 0
        sf = self.sets[0].scale_factor
        scale = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        flip = random.random() <= 0.5

        input, target, weight, meta = [], [], [], []

        for k, data in enumerate(self.sets):   
            # loop the small dataset to construct pair
            if k==0 or k == 1:
                t_idx = self.data_ind[idx]
            else:
                t_idx = idx

            # Get the same transform for unsup and unsup_aug
            if k==0 or k == 1:
                i, t, w, m = data.__getitem__(t_idx)
            else:
                i, t, w, m = data.__getitem__(t_idx, rot_angle = rot, scale = scale, flip=flip)
                
            input.append(i)

            # For unsup dataset, set full vis  
            if k == 2:
                w.zero_()
                w[ai_joint_ind, :] = 1
                m['joints_vis'] = meta[0]['joints_vis'].copy()
                m['joints_vis'][ai_joint_ind, :] = 1

            if k<=2:
                target.append(t)
                weight.append(w)
                meta.append(m)


        return input, target, weight, meta