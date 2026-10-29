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



# base on the results of FVIA
class GetTrainImageDirectly(Dataset):
    def __init__(self, path_db=None, path_csv=None, transform=None, csv_name=None, postfix=None, camera=None, angle=1, mode=None, th=None, fixed_light=None):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.camera = camera
        self.angle = angle
        self.mode = mode
        self.th = th
        self.fixed_light = fixed_light

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        id = self.csv['finger'][index]
        if self.camera is not None:
            if self.camera == 1:
                lc = 'lc1'
            elif self.camera == 2:
                lc = 'lc2'
            elif self.camera == 3:
                lc = 'lc3'
            else:
                print('camera error!')
            img = str(self.csv['finger'][index]) + '_' + str(self.angle) + '_'  + str(self.csv[lc][index])  + '_' + str(self.camera) + '_' + str(self.csv['sample'][index])
            path_img = self.path_db + '/' + img + self.postfix
            img = Image.open(path_img)#.convert('L')
            img = np.array(img)
            if self.transform:
                img = self.transform(img)
            img_1 = img
            img_2 = img
            img_3 = img
            # return img, id # 
        else:
            img_1 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc1'][index]) + '_1' + '_' + str(self.csv['sample'][index])
            img_2 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc2'][index]) + '_2' + '_' + str(self.csv['sample'][index])
            img_3 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc3'][index]) + '_3' + '_' + str(self.csv['sample'][index])
            path_img_1 = self.path_db + '/' + img_1 + self.postfix
            path_img_2 = self.path_db + '/' + img_2 + self.postfix
            path_img_3 = self.path_db + '/' + img_3 + self.postfix
            img_1 = Image.open(path_img_1).convert('RGB')
            img_2 = Image.open(path_img_2).convert('RGB')
            img_3 = Image.open(path_img_3).convert('RGB')
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_3 = np.array(img_3)
            if self.transform:
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                img_3 = self.transform(img_3)
                # print(id)
        return img_1, img_2, img_3, id


# base on the results of FVIA
class GetAuthenticationPairDirectly(Dataset):
    def __init__(self, path_db=None, path_csv = None, transform = None, csv_name = None, postfix = '.bmp', camera=None, angle=1, mode=None, th=None, fixed_light=None):
        self.path_db = path_db
        self.transform = transform
        self.csv_name = csv_name
        self.postfix = postfix
        self.path_csv = os.path.join(sys.path[0], path_csv, self.csv_name)
        self.csv = pd.read_csv(self.path_csv)
        self.camera = camera
        self.angle = angle
        self.mode = mode
        self.th = th
        self.fixed_light = fixed_light

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        bio_ref_subject_id = self.csv['finger_enrolled'][index]
        probe_subject_id = self.csv['finger_probe'][index]
        label = self.csv['label'][index]
        if self.camera is not None:
            if self.camera == 1:
                lc = 'lc1'
            elif self.camera == 2:
                lc = 'lc2'
            elif self.camera == 3:
                lc = 'lc3'
            else:
                print('camera error!')
            bio_ref_reference_id = str(self.csv['finger_enrolled'][index]) + '_' + str(self.angle) + '_' + str(self.csv[lc][index]) + '_' + str(self.camera) + '_' + str(self.csv['sample_enrolled'][index])
            probe_reference_id = str(self.csv['finger_probe'][index]) + '_' + str(self.angle) + '_' + str(self.csv[lc][index]) + '_' + str(self.camera) + '_' + str(self.csv['sample_probe'][index])
            path_img1 = self.path_db + '/' + bio_ref_reference_id + self.postfix
            path_img2 = self.path_db + '/' + probe_reference_id + self.postfix
            img1 = Image.open(path_img1).convert('RGB')
            img1 = np.array(img1)
            img2 = Image.open(path_img2).convert('RGB')
            img2 = np.array(img2)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img1_1 = img1
            img1_2 = img1
            img1_3 = img1
            img2_1 = img2
            img2_2 = img2
            img2_3 = img2
        else:
            bio_ref_reference_id_1 = str(self.csv['finger_enrolled'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc1_enrolled'][index]) + '_1_' + str(self.csv['sample_enrolled'][index])
            bio_ref_reference_id_2 = str(self.csv['finger_enrolled'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc2_enrolled'][index]) + '_2_' + str(self.csv['sample_enrolled'][index])
            bio_ref_reference_id_3 = str(self.csv['finger_enrolled'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc3_enrolled'][index]) + '_3_' + str(self.csv['sample_enrolled'][index])
            probe_reference_id_1 = str(self.csv['finger_probe'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc1_probe'][index]) + '_1_' + str(self.csv['sample_probe'][index])
            probe_reference_id_2 = str(self.csv['finger_probe'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc2_probe'][index]) + '_2_' + str(self.csv['sample_probe'][index])
            probe_reference_id_3 = str(self.csv['finger_probe'][index]) + '_' + str(self.angle) + '_' + str(self.csv['lc3_probe'][index]) + '_3_' + str(self.csv['sample_probe'][index])
            path_img1_1 = self.path_db + '/' + bio_ref_reference_id_1 + self.postfix
            path_img1_2 = self.path_db + '/' + bio_ref_reference_id_2 + self.postfix
            path_img1_3 = self.path_db + '/' + bio_ref_reference_id_3 + self.postfix
            path_img2_1 = self.path_db + '/' + probe_reference_id_1 + self.postfix
            path_img2_2 = self.path_db + '/' + probe_reference_id_2 + self.postfix
            path_img2_3 = self.path_db + '/' + probe_reference_id_3 + self.postfix
            img1_1 = Image.open(path_img1_1).convert('RGB')
            img1_2 = Image.open(path_img1_2).convert('RGB')
            img1_3 = Image.open(path_img1_3).convert('RGB')
            img2_1 = Image.open(path_img2_1).convert('RGB')
            img2_2 = Image.open(path_img2_2).convert('RGB')
            img2_3 = Image.open(path_img2_3).convert('RGB')
            img1_1 = np.array(img1_1)
            img1_2 = np.array(img1_2)
            img1_3 = np.array(img1_3)
            img2_1 = np.array(img2_1)
            img2_2 = np.array(img2_2)
            img2_3 = np.array(img2_3)
            if self.transform:
                img1_1 = self.transform(img1_1)
                img1_2 = self.transform(img1_2)
                img1_3 = self.transform(img1_3)
                img2_1 = self.transform(img2_1)
                img2_2 = self.transform(img2_2)
                img2_3 = self.transform(img2_3)
        return img1_1, img1_2, img1_3, img2_1, img2_2, img2_3, label, probe_subject_id, bio_ref_subject_id


