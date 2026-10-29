#!/bin/bash
## Sun's Grid Engine parameters
# ... path to the conda environment python executable
#$ -S /idiap/temp/jhuang/conda/conda3/envs/bob/bin/python3
# ... queue flags you want to use, this is just an example if you want PyTorch and SGPU
#$ -l pytorch,lgpu,'hostname=vgn[j]*'
# ... job name
#$ -N fdt
# ... project name (this is apparently your project name: oh-sm)
#$ -P oh-sm
# ... Path to existing directories for error logs and output logs from SGE
#$ -o /idiap/temp/jhuang/code/FDT_code/FDT/logs
#$ -e /idiap/temp/jhuang/code/FDT_code/FDT/logs
# ... You may need to add cwd to be able to reference local files in the directory where your main.py is located, in which case you need to have:
#$ -cwd

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
sys.path.append(os.getcwd())
import argparse
SCRIPTS_DIR = os.getcwd()
sys.path.append(SCRIPTS_DIR)
import torch
import torch.optim as optim
import torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from loss.CenterLoss import CenterLoss
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random
from labour.Fit import fit
from dataload.mffv_dataload import GetTrainImageDirectly, GetAuthenticationPairDirectly
from dataload.lfmb3dfb_dataload import GetTrainImage, GetAuthenticationPair
from model.fdt.fdt import FDT, BAcos, BatchFormer, BA
from model.mvcnn.mvcnn_ori import MVCNN_ori
from model.mvcnn.mvcnn_imp import MVCNN_imp
from model.mvt.mvt_ori import MVT_ori
from model.mvt.mvt_imp import MVT_imp




# # ##################################################seting############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='fdt', help='model and mode', required=True)
parser.add_argument('-d', '--dataset', type=str, default='MFFV-N', help='name of dataset', required=True)
parser.add_argument('-db', '--db_path', type=str, default='E:/FV/MFFV-N/', help='db path', required=True)
parser.add_argument('-pt', '--protocol_path', type=str, default='pt/', help='pt path', required=True)
parser.add_argument('-ep', '--epochs', type=int, default=5000, help='training epoch', required=True)
parser.add_argument('-rs', '--results', type=str, default='rs/', help='model save path', required=True)
args = parser.parse_args()

####################################################for fix random seed################################################
seed = 2023
os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
def seed_torch(seed):
	  random.seed(seed)
	  os.environ['PYTHONHASHSEED'] = str(seed) 
	  np.random.seed(seed)
	  torch.manual_seed(seed)
	  torch.cuda.manual_seed(seed)
	  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
seed_torch(seed)
def worker_init_fn(worker_id):
    np.random.seed(seed)
    random.seed(seed+worker_id)
g = torch.Generator()
g.manual_seed(seed)

## for data augmentation
train_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((240, 320)),
])

test_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((240, 320)),
])


# for dataloader
if args.dataset == 'MFFV-N':
    train_dataset = GetTrainImageDirectly(path_db=args.db_path, path_csv=args.protocol_path, csv_name='train_after_ia.csv', postfix='.bmp', transform=train_data_transform)
    dev_dataset = GetAuthenticationPairDirectly(path_db=args.db_path, path_csv=args.protocol_path, csv_name='mffv_fvia_dev_bal.csv', postfix='.bmp', transform=test_data_transform)
elif args.dataset == 'LFMB-3DFB':
    train_dataset = GetTrainImage(path_db=args.db_path, path_csv=args.protocol_path, csv_name='lfmb3dfb_train.csv', postfix='.bmp', transform=test_data_transform, camera=['AA', 'CC', 'EE'])
    dev_dataset = GetAuthenticationPair(path_db=args.db_path, path_csv=args.protocol_path, csv_name='lfmb3dfb_dev.csv', postfix='.bmp', transform=test_data_transform, camera=['AA', 'CC', 'EE'])
else:
    print('please provide dataset name!')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, generator=g)
dev_loader = torch.utils.data.DataLoader(dataset = dev_dataset, batch_size=32, shuffle = False, num_workers = 8, worker_init_fn=worker_init_fn, generator=g)

# for losses
ce_loss = nn.CrossEntropyLoss()
# ct_loss = CenterLoss(num_classes=1000, feat_dim=feat_dim)
if args.model[:3] == 'fdt':
    ct_loss = CenterLoss(num_classes=1000, feat_dim=128)
elif args.model[:3] == 'mvt':
    ct_loss = CenterLoss(num_classes=1000, feat_dim=384)
elif args.model == 'mvcnn_ori':
    ct_loss = CenterLoss(num_classes=1000, feat_dim=4096)
elif args.model == 'mvcnn_imp':
    ct_loss = CenterLoss(num_classes=1000, feat_dim=512)
else:
    print('some thing wrong with the model name!')

############################################################## model################################################################
if args.model == 'fdt':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
    use_ba=True, batch_train=BA,
    use_lffn=True,
    use_mlppatch=True,
    use_peg=True)
    use_ba = True
elif args.model == 'fdt_wo_ba':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
    use_ba=False, batch_train=BA,
    use_lffn=True,
    use_mlppatch=True,
    use_peg=True)
    use_ba = False
elif args.model == 'fdt_wo_lf':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
    use_ba=True, batch_train=BA,
    use_lffn=False,
    use_mlppatch=True,
    use_peg=True)
    use_ba = True
elif args.model == 'fdt_wo_mp':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
    use_ba=True, batch_train=BA,
    use_lffn=True,
    use_mlppatch=False,
    use_peg=True)
    use_ba = True
elif args.model == 'fdt_wo_pe':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
    use_ba=True, batch_train=BA,
    use_lffn=True,
    use_mlppatch=True,
    use_peg=False)
    use_ba = True
elif args.model == 'fdt_wo_dr':
    model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(300/300, 300/300, 300/300, 300/300, 300/300, 300/300), keep_rate_sv=(100/100, 100/100, 100/100, 100/100), fuse_token=True, use_fa=False, 
    use_ba=True, batch_train=BA, 
    use_lffn=True, 
    use_mlppatch=True,
    use_peg=True)
    use_ba = True
elif args.model == 'mvcnn_ori':
    model = MVCNN_ori()
    use_ba = False
elif args.model == 'mvcnn_imp':
    model = MVCNN_imp()
    use_ba = False
elif args.model == 'mvt_ori':
    model = MVT_ori()
    use_ba = False
elif args.model == 'mvt_imp':
    model = MVT_imp()
    use_ba = False
else:
    print('some wrong with the model name!')


# for optimizer
lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)#
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.00000001, last_epoch=-1) #

def check_files(folder_path, extension):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            return True
    return False

def get_max_file(folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    return files[-1]

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('running with gpu')
        ce_loss.cuda()
        ct_loss.cuda()
        model.cuda()
    else:
        print('running with cpu')
        
    if check_files(args.results+'continue_model/', '.ckpt'):
        print('there are some ckpt')
        last_ckpt = get_max_file(args.results+'continue_model/')
        model.load_state_dict(torch.load(args.results+'continue_model/'+last_ckpt))
        last_epoch = int(last_ckpt[:-5])
        print('stating with epoch:', last_epoch)
    else:
        print('there is no ckpt')
        last_epoch = 0

    fit(args=args,
    train_loader=train_loader,
    dev_loader=dev_loader,
    model=model,
    loss_fn1=ce_loss,
    beta=0.01,
    loss_fn2=ct_loss,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=args.epochs-last_epoch,
    last_epoch = last_epoch,
    model_save_path=args.results,
    print_label=args.model,
    use_ba=use_ba)

# qsub train.py -m fdt -d MFFV-N -db /idiap/resource/database/MFFV-N/data/ -pt /idiap/temp/jhuang/code/FDT_code/PT/ -ep 5000 -rs RS/trained_models/fdt_mffv/
# qsub train.py -m fdt -d LFMB-3DFB -db /idiap/resource/database/SCUT_LFMB-3DFB/LFMB-3DFB_Pictures_Seged_Rectified/ -pt /idiap/temp/jhuang/code/FDT_code/PT/ -ep 5000 -rs RS/trained_models/fdt_lfmb3dfb/

    