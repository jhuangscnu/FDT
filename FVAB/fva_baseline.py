import os
import sys
import argparse
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
from bob.bio.vein.extractor import MaximumCurvature
from bob.bio.vein.extractor import RepeatedLineTracking
from bob.bio.vein.extractor import WideLineDetector
from bob.bio.base.transformers import PreprocessorTransformer
from bob.bio.vein.preprocessor import NoCrop, NoMask, NoNormalization, NoFilter, Preprocessor
from bob.bio.vein.algorithm.MiuraMatch import MiuraMatch


class GetAuthenticationPair(Dataset):
    def __init__(self, path_fe=None, path_csv = None, prefix = None, csv_name = None, camera=None, postfix = None):
        self.path_fe = path_fe
        self.path_csv = path_csv
        self.csv_name = csv_name
        self.prefix = prefix
        self.postfix = postfix
        self.camera = camera
        pwd = sys.path[0]
        csv_file_dir = os.path.join(pwd, self.path_csv)
        test_file_path = os.path.join(csv_file_dir, self.csv_name)  
        self.test_df = pd.read_csv(test_file_path)
    
    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index): 
        probe_reference_id = self.test_df['name1'][index]
        probe_subject_id = self.test_df['name1'][index]
        probe_subject_id = probe_subject_id[:6]
        
        bio_ref_reference_id = self.test_df['name2'][index]
        bio_ref_subject_id = self.test_df['name2'][index]
        bio_ref_subject_id = bio_ref_subject_id[:6]
        
        path_img1 = self.path_fe + self.prefix + '_' + self.test_df['name1'][index] + '_' + self.camera + self.postfix
        path_img2 = self.path_fe + self.prefix + '_' + self.test_df['name2'][index] + '_' + self.camera + self.postfix
        
        img1 = np.load(path_img1)
        img2 = np.load(path_img2)
        
        label = self.test_df['label'][index]
        
        return img1, img2, label, probe_reference_id, probe_subject_id, bio_ref_reference_id, bio_ref_subject_id


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_rs_csv(csv_data, csv_path, csv_name, csv_title):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    path_csv = csv_path + csv_name
    csv = pd.DataFrame(columns=csv_title, data=csv_data)
    csv.to_csv(path_csv, index=False)




def run(path_fe=None, path_csv=None, path_save=None, save_csv=None, csv_name=None, camera=None, fem='mc', postfix='.npy'):
    print('running for: ', csv_name)
    title = ['probe_reference_id', 'probe_subject_id', 'bio_ref_reference_id', 'bio_ref_subject_id', 'score']
    dataset = GetAuthenticationPair(path_fe=path_fe, path_csv=path_csv, csv_name = csv_name, prefix = fem, camera = camera, postfix = postfix)
    data_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)
    miura_match = MiuraMatch()
    result_data = []
    print('running baseline ', fem, 'for camera ', camera)

    for img1, img2, label, probe_reference_id, probe_subject_id, bio_ref_reference_id, bio_ref_subject_id in tqdm(data_loader):
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = img1.numpy()
        img2 = img2.numpy()
        score = miura_match.score(model=img1, probe=img2)
        result_data.append([probe_reference_id[0], probe_subject_id[0], bio_ref_reference_id[0], bio_ref_subject_id[0], score])

    save_rs_csv(csv_data=result_data, csv_path=path_save, csv_name=save_csv, csv_title=title)


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-fe','--path_fe', type=str, help='path of database', required=True)
    parser.add_argument('-pt','--path_pt', type=str, help='path of protocols', required=True)
    parser.add_argument('-rs','--path_rs', type=str, help='path of results', required=True)
    args = parser.parse_args()

    for c in ['AA', 'CC', 'EE']:
        # balanced, dev
        save_csv = 'dev_balance_' + str(c) + '.csv'
        print('for: ', save_csv)
        run(path_fe=args.path_fe, path_csv=args.path_pt, csv_name='lfmb3dfb_dev.csv', path_save=args.path_rs, save_csv=save_csv, camera=c)
        # balanced, test
        save_csv = 'test_balance_' + str(c) + '.csv'
        print('for: ', save_csv)
        run(path_fe=args.path_fe, path_csv=args.path_pt, csv_name='lfmb3dfb_test.csv', path_save=args.path_rs, save_csv=save_csv, camera=c)
        

if __name__ == '__main__':
    main(sys.argv)
    
# python fva_baseline.py -fe /idiap/temp/jhuang/dataset/LFMB-3DFB_mc_npy/ -pt /idiap/temp/jhuang/code/FDT_code/PT/ -rs RS/

