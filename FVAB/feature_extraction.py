import os
import sys
from PIL import Image
import argparse
from pathlib import Path
import numpy as np
from bob.bio.vein.extractor import MaximumCurvature
from bob.bio.base.transformers import PreprocessorTransformer
from bob.bio.vein.preprocessor import NoCrop, NoMask, NoNormalization, NoFilter, Preprocessor
import pandas as pd



prep = Preprocessor(crop=NoCrop(), mask=NoMask(), normalize=NoNormalization(), filter=NoFilter())
fe = MaximumCurvature()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def FE(name_imgs, path_fe, path_db):
    name_imgs = name_imgs
    num_imgs = len(name_imgs)
    for i, name_img in enumerate(name_imgs):
        name_f = 'mc_' + os.path.splitext(name_img)[0]
        path_f = os.path.join(path_fe, name_f)
        my_outf = path_f+'.npy'
        if os.path.isfile(my_outf):
            print('Nothing to do; feature-file already exists')
        else:
            #path_f = path_f[0:-4] + '.npy'
            #if not os.path.exists(path_f):
            path_img = os.path.join(path_db, name_img)
            img = Image.open(path_img).convert('L')
            img = img.resize((320, 240))
            img = np.array(img)
            pre_img = prep(img)
            f = fe(pre_img) #*255
            np.save(path_f, f)
            #else:
            #    print('feature file exists, nothing to do')
        print(i+1, '/', num_imgs)
    
    return

def get_list(pt):
    finger_dev_test = []
    list_dev_test = []
    for file in ['lfmb3dfb_dev.csv', 'lfmb3dfb_test.csv']:
        df = pd.read_csv(pt+file)
        finger_1 = list(set(df['name1']))
        finger_2 = list(set(df['name2']))
        finger_dev_test = finger_dev_test + list(set(finger_1 + finger_2))
    for finger in finger_dev_test:
        for view in ['AA', 'CC', 'EE']:
            list_dev_test.append(finger + '_' + view + '.bmp')

    return list_dev_test

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--path_db', type=str, help='path of database', required=True)
    parser.add_argument('-fe','--path_fe', type=str, help='path of feature', required=True)
    parser.add_argument('-pt','--path_pt', type=str, help='path of protocol', required=True)
    args = parser.parse_args()

    path_db = args.path_db
    path_fe = args.path_fe
    
    #mkdir(path_fe)
    if not os.path.exists(path_fe):
        os.makedirs(path_fe)
    name_imgs = sorted(get_list(args.path_pt))
    
    #exclude files from path_db that do not end in .bmp .
    fv_files = []
    for f in name_imgs:
        if f.endswith('.bmp'):
            fv_files.append(f)
    
    num_files = len(fv_files)
    print('Total number of fingervein files to process:', num_files)
    
    
    # make chunks of 10 files; process each chunk as 1 job on grid
    chunk_size = 10
    chunks = [fv_files[x:x+chunk_size] for x in range(0, len(fv_files), chunk_size)]  
    num_chunks = len(chunks)
    print('Number of chunks:', num_chunks)
    
    # if idiap grid is available use it to distribute tasks
    task_id = 0
    num_tasks = 1
    if 'SGE_TASK_ID' in os.environ and os.environ['SGE_TASK_ID'] != 'undefined':
        task_id = int(os.environ['SGE_TASK_ID'])
        num_tasks = int(os.environ['SGE_TASK_LAST'])
        print('SGE_stats:', task_id, num_tasks)

        if task_id > num_chunks:
            assert 0, 'Grid request for job %d on a setup with %d jobs' % (task_id, num_chunks)
        file_set = chunks[task_id-1]
    else:
        # idiap grid not being used
        file_set = fv_files

    print('Num. fingervein files:', len(file_set))
    
    FE(name_imgs=file_set, path_fe=path_fe, path_db=path_db)



if __name__ == '__main__':
    main(sys.argv)

# python feature_extraction.py -db /idiap/resource/database/SCUT_LFMB-3DFB/LFMB-3DFB_Pictures_Seged_Rectified/ -fe /idiap/temp/jhuang/dataset/LFMB-3DFB_mc_npy/ -pt /idiap/temp/jhuang/code/FDT_code/PT/
