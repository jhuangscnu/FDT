from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import argparse
import sys

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cumsum(img, idx):
    if idx < 3:
        img = rearrange(img, '(h1 h) (w1 w) -> (h1 w1) h w', h1=10, w1=10)
    else:
        img = rearrange(img, '(h1 h) (w1 w) -> (h1 w1) h w', h1=30, w1=10)
    img = rearrange(img, 'n h w -> n (h w)')
    cov = np.cov(img)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    tot = sum(eig_vals)
    eig_vals_nom = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_eig_vals_nom = np.cumsum(eig_vals_nom)

    return cum_eig_vals_nom

class GetTrainImageDirectly(Dataset):
    def __init__(self, path_db=None, path_csv=None, transform=None, csv_name=None, postfix=None, camera=None, angle=None, mode=None, th=None, fixed_light=None):
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
        if (index%10)==1:
            img_1 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc1'][index]) + '_1' + '_' + str(self.csv['sample'][index])
            img_2 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc2'][index]) + '_2' + '_' + str(self.csv['sample'][index])
            img_3 = str(self.csv['finger'][index]) + '_' + str(self.angle)  + '_' + str(self.csv['lc3'][index]) + '_3' + '_' + str(self.csv['sample'][index])
            path_img_1 = self.path_db + '/' + img_1 + self.postfix
            path_img_2 = self.path_db + '/' + img_2 + self.postfix
            path_img_3 = self.path_db + '/' + img_3 + self.postfix
            img_1 = Image.open(path_img_1).convert('L')
            img_2 = Image.open(path_img_2).convert('L')
            img_3 = Image.open(path_img_3).convert('L')
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_3 = np.array(img_3)

            return img_1, img_2, img_3, id
        else:
            return 0, 0, 0, 0

def reach_99(info_dist):
    num_99 = 0
    for cum in info_dist:
        num_99 += 1
        if cum > 99:
            # print(num)
            break
    return num_99


def run(path_db=None, path_csv=None, csv_name=None, postfix=None, transform=None, camera=None, angle=None, mode=None, th=None, fixed_light=None):
    dataset = GetTrainImageDirectly(path_db=path_db, path_csv=path_csv, transform = transform, csv_name = csv_name, postfix = postfix, camera=camera, angle=angle, mode=mode, th=th, fixed_light=fixed_light)
    data_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)

    list_all_samples_num_99 = [] # for storing statistics for all samples
    for img_1, img_2, img_3, id in tqdm(data_loader):
        if id: # because only the first sample return no zero (sampling), see class GetTrainImageDirectly
            img_1 = torch.squeeze(img_1, 0)
            img_2 = torch.squeeze(img_2, 0)
            img_3 = torch.squeeze(img_3, 0)
            img_123 = np.concatenate([img_1, img_2, img_3], axis=0)

            list_one_sample_num_99 = [] # for storing statistics for one samples
            for idx, img in enumerate([img_1, img_2, img_3, img_123]):
                info_dist = cumsum(img, idx)
                num_99 = reach_99(info_dist)
                list_one_sample_num_99.append(num_99)
            list_all_samples_num_99.append(list_one_sample_num_99)

    return list_all_samples_num_99


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--path_db', type=str, help='path of database', required=True)
    parser.add_argument('-pt','--path_pt', type=str, help='path of protocols', required=True)
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True)
    args = parser.parse_args()

    list_all_samples_num_99 = run(path_db=args.path_db, path_csv=args.path_pt, csv_name='train_after_ia.csv', postfix='.bmp', transform=None, camera=None, angle=1, mode=None, th=None, fixed_light=None)
    list_all_samples_num_99 = list(map(list, zip(*list_all_samples_num_99)))

    mkdir(args.path_rs)
    plt.boxplot(list_all_samples_num_99,
            patch_artist=True,
            boxprops={'color': 'blue'},
            labels=['View 1', 'View 2', 'View 3', 'Full-View'])
    plt.ylabel('Minimum of components contain 99% information', size=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(args.path_rs+'ra_box.png')


if __name__ == '__main__':
    main(sys.argv)

# python general_statistic.py -db /path/of/the/MFFV/dataset/ -pt /path/of/the/PT/ -rs RS/
