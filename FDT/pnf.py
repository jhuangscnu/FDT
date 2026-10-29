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

import torch.nn as nn
from thop import profile
import torch
import argparse
import os

from model.fdt.fdt import FDT, BAcos, BatchFormer, BA
from model.mvcnn.mvcnn_ori import MVCNN_ori
from model.mvcnn.mvcnn_imp import MVCNN_imp
from model.mvt.mvt_ori import MVT_ori
from model.mvt.mvt_imp import MVT_imp

# # ##################################################seting############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-rs', '--results', type=str, default='params_flops/', help='model save path', required=False)
parser.add_argument('-device', '--device', type=str, default='6', help='gpu index', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

[fdt, mvcnn_ori, mvcnn_imp, mvt_ori, mvt_imp] = [FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False, use_ba=True, batch_train=BA, use_lffn=True, use_mlppatch=True, use_peg=True), MVCNN_ori(), MVCNN_imp(), MVT_ori(), MVT_imp()]

[fdt_wo_pe, fdt_wo_mp, fdt_wo_lf, fdt_wo_dr, fdt_wo_ba] = [FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False, use_ba=True, batch_train=BA,use_lffn=True,use_mlppatch=True,use_peg=False),FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,use_ba=True, batch_train=BA,use_lffn=True,use_mlppatch=False,use_peg=True),FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,use_ba=True, batch_train=BA,use_lffn=False,use_mlppatch=True,use_peg=True), FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(300/300, 300/300, 300/300, 300/300, 300/300, 300/300), keep_rate_sv=(100/100, 100/100, 100/100, 100/100), fuse_token=True, use_fa=False, use_ba=True, batch_train=BA, use_lffn=True, use_mlppatch=True,use_peg=True),FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,use_ba=False, batch_train=BA,use_lffn=True,use_mlppatch=True,use_peg=True)]


def get_params_flops(list_model=None, list_name=None, path_rs=None, name_txt=None):
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    list_rs = []
    x = torch.randn(1, 1, 240, 320)
    x = [x, x, x]
    msg = ''
    for i, model in enumerate(list_model):
        print('model name:', list_name[i])
        flops, params = profile(model, inputs=(x,))
        
        if model == fdt:
            f = torch.randn(1, 128)
        elif model == mvcnn_ori:
            f = torch.randn(1, 4096)
        elif model == mvcnn_imp:
            f = torch.randn(1, 512)
        elif model == mvt_ori:
            f = torch.randn(1, 384)
        elif model == mvt_imp:
            f = torch.randn(1, 384)
        elif model == fdt_wo_pe:
            f = torch.randn(1, 128)
        elif model == fdt_wo_dr:
            f = torch.randn(1, 128)
        elif model == fdt_wo_mp:
            f = torch.randn(1, 128)
        elif model == fdt_wo_lf:
            f = torch.randn(1, 128)
        elif model == fdt_wo_ba:
            f = torch.randn(1, 128)
        else:
            print('no model name')
            
        if model == mvcnn_imp:
            flops_cls, params_cls = profile(model.net2, inputs=(f,))
        elif model == fdt:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        elif model == fdt_wo_pe:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        elif model == fdt_wo_dr:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        elif model == fdt_wo_mp:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        elif model == fdt_wo_lf:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        elif model == fdt_wo_ba:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
            flops_ba, params_ba = profile(model.ba, inputs=(f,))
            flops_cls = flops_cls + flops_ba
            params_cls = params_cls + params_ba
        else:
            flops_cls, params_cls = profile(model.f2c, inputs=(f,))
       
        msg += '\n' + list_name[i] + ' params: ' + str((params-params_cls)/1e6) + 'M, flops: ' + str((flops-flops_cls)/1e9) + 'G'
    with open(os.path.join(path_rs, name_txt), 'w') as f:
        f.write(msg)
        

get_params_flops(list_model=[fdt, mvcnn_ori, mvcnn_imp, mvt_ori, mvt_imp], list_name=['fdt', 'mvcnn_ori', 'mvcnn_imp', 'mvt_ori', 'mvt_imp'], path_rs=args.results, name_txt='fdt_comparison.txt')
get_params_flops(list_model=[fdt_wo_pe, fdt_wo_mp, fdt_wo_lf, fdt_wo_dr, fdt_wo_ba],  list_name=['fdt_wo_pe', 'fdt_wo_mp', 'fdt_wo_lf', 'fdt_wo_dr', 'fdt_wo_ba'] , path_rs=args.results, name_txt='fdt_ablation.txt')


