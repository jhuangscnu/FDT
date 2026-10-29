from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.pyplot as plt
import sys
import argparse
import os

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

def information_distributions(db=None, rs=None):
    # just take sample 1 as example
    img1 = Image.open(db+'1_1_6_1_1.bmp')
    img2 = Image.open(db+'1_1_6_2_1.bmp')
    img3 = Image.open(db+'1_1_5_3_1.bmp')

    img1 = np.array(img1)
    img2 = np.array(img2)
    img3 = np.array(img3)
    img123 = np.concatenate([img1, img2, img3], axis=0)

    for idx, img in enumerate([img1, img2, img3, img123]):
        info_dist = cumsum(img, idx)
        # plt.subplot()
        if idx < 3:
            plt.figure(figsize=(7.2, 6))
            plt.bar(range(0, 100), info_dist)
            threshold_idx = np.argmax(info_dist > 99)
            if threshold_idx > 0:  
                plt.axvline(x=threshold_idx, color='red', linestyle='--', label=f'99% at PC {threshold_idx+1}')
        else:
            plt.figure(figsize=(24, 6))
            plt.bar(range(0, 300), info_dist)
            threshold_idx = np.argmax(info_dist > 99)
            if threshold_idx > 0:  # 确保找到了有效的索引
                plt.axvline(x=threshold_idx, color='red', linestyle='--', label=f'99% at PC {threshold_idx+1}')
        plt.xlabel('Percentage of PCs (%)', fontdict={'size':22})
        plt.xticks(size=18)
        plt.ylabel('Cumulative information (%)', fontdict={'size':22})
        plt.yticks(size=18)
        plt.savefig(rs + 'info_dist_img_'+str(idx+1)+'.png')

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--path_db', type=str, help='path of database', required=True)
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True)
    args = parser.parse_args()

    mkdir(args.path_rs)
    information_distributions(db=args.path_db, rs=args.path_rs)


if __name__ == '__main__':
    main(sys.argv)

# python information_distribution.py -db /path/of/the/MFFV/dataset/ -rs RS/
