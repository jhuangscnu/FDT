"""
===============================================================================
Author(Copyright (C) 2025): Junduan Huang, Sushil Bhattacharjee, Sébastien Marcel, Wenxiong Kang.
Institution: 
School of Artificial Intelligence at South China Normal University, Foshan, 528225, China;
School of Automation Science and Engineering at South China University of Technology, Guangzhou, 510641, China;
Biometrics Security and Privacy Group at Idiap Research Institute, Martigny, 1920, Switzerland.

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at runrunjun@163.com
===============================================================================
"""

import os
import sys
from PIL import Image
from PIL import ImageDraw
import cv2 as cv
import numpy as np
import math
import shutil
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm



class GetTrainImage(Dataset):
    def __init__(self, path_db=None, path_csv=None, transform=None, csv_name=None, postfix=None, camera=None):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.camera = camera

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        ref_id = self.csv['name'][index]
        id = int(ref_id[:4])
        if id >2000:
            id = id - 1374
        img = []
        for c in self.camera:
            ref_id_img = ref_id + '_' + c
            path_ref_id_img = self.path_db + '/' + ref_id_img + self.postfix
            img_ref_id = Image.open(path_ref_id_img).convert('RGB')
            img_ref_id = np.array(img_ref_id)
            if self.transform:
                img_ref_id = self.transform(img_ref_id)
            img.append(img_ref_id)
        return img[0], img[1], img[2], id


class GetAuthenticationPair(Dataset):
    def __init__(self, path_db=None, path_csv = None, transform = None, csv_name = None, postfix = '.bmp', camera=None):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(sys.path[0], path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.camera = camera

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        bio_ref_subject_id = self.csv['name1'][index]
        id1 = int(bio_ref_subject_id[:4])
        if id1 >2000:
            id1 = id1 - 1374

        probe_subject_id = self.csv['name2'][index]
        id2 = int(probe_subject_id[:4])
        if id2 >2000:
            id2 = id2 - 1374

        label = self.csv['label'][index]

        img1 = []
        img2 = []
        for c in self.camera:
            ref_id_img1 = bio_ref_subject_id + '_' + c
            path_ref_id_img1 = self.path_db + '/' + ref_id_img1 + self.postfix
            img1_ref_id = Image.open(path_ref_id_img1).convert('RGB')
            img1_ref_id = np.array(img1_ref_id)
            if self.transform:
                img1_ref_id = self.transform(img1_ref_id)
            img1.append(img1_ref_id)

            ref_id_img2 = probe_subject_id + '_' + c
            path_ref_id_img2 = self.path_db + '/' + ref_id_img2 + self.postfix
            img2_ref_id = Image.open(path_ref_id_img2).convert('RGB')
            img2_ref_id = np.array(img2_ref_id)
            if self.transform:
                img2_ref_id = self.transform(img2_ref_id)
            img2.append(img2_ref_id)


        return img1[0], img1[1], img1[2], img2[0], img2[1], img2[2], label, id1, id2

