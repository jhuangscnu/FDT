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
import pandas as pd
from dataload.mffv_dataload import GetAuthenticationPairDirectly
from labour.EpochOperation import dev_epoch
from torchvision import transforms
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import numpy
import argparse

from model.fdt.fdt import FDT, BAcos, BatchFormer, BA
from model.mvcnn.mvcnn_ori import MVCNN_ori
from model.mvcnn.mvcnn_imp import MVCNN_imp
from model.mvt.mvt_ori import MVT_ori
from model.mvt.mvt_imp import MVT_imp



# # ##################################################seting############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=list, default=['fdt_wo_pe_mffv', 'fdt_wo_mp_mffv', 'fdt_wo_lf_mffv', 'fdt_wo_dr_mffv', 'fdt_wo_ba_mffv'], help='model list', required=False) #  ['fdt_mffv', 'mvcnn_ori_mffv_lgpu', 'mvcnn_imp_mffv', 'mvt_ori_mffv', 'mvt_imp_mffv']
parser.add_argument('-rs', '--results', type=str, default='SNM/', help='metrics save path', required=False)
parser.add_argument('-ds', '--ds_type', type=str, default='tex', help='dataset type', required=False)
parser.add_argument('-db', '--db_path', type=str, default='/home/data/dataset/FV/MFFV-N/', help='db path', required=False)
parser.add_argument('-device', '--device', type=str, default='6', help='gpu index', required=False)
parser.add_argument('-pt', '--pt_path', type=str, default='/home/hjd/code/FDT_code/PT/', help='protocol path', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
#########################################################################################################################

test_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((240, 320)),
])
                                              
bal_dev_dataset = GetAuthenticationPairDirectly(path_db=args.db_path, path_csv=args.pt_path, csv_name='mffv_fvia_dev_bal.csv', postfix='.bmp', transform=test_data_transform)
bal_dev_loader = torch.utils.data.DataLoader(dataset = bal_dev_dataset, batch_size=8, shuffle=False, num_workers = 8)

bal_test_dataset = GetAuthenticationPairDirectly(path_db=args.db_path, path_csv=args.pt_path, csv_name='mffv_fvia_test_bal.csv', postfix='.bmp', transform=test_data_transform)
bal_test_loader = torch.utils.data.DataLoader(dataset = bal_test_dataset, batch_size=8, shuffle=False, num_workers = 8)


nom_dev_dataset = GetAuthenticationPairDirectly(path_db=args.db_path, path_csv=args.pt_path, csv_name='mffv_fvia_dev_nom.csv', postfix='.bmp', transform=test_data_transform)
nom_dev_loader = torch.utils.data.DataLoader(dataset = nom_dev_dataset, batch_size=8, shuffle=False, num_workers = 8)

nom_test_dataset = GetAuthenticationPairDirectly(path_db=args.db_path, path_csv=args.pt_path, csv_name='mffv_fvia_test_nom.csv', postfix='.bmp', transform=test_data_transform)
nom_test_loader = torch.utils.data.DataLoader(dataset = nom_test_dataset, batch_size=8, shuffle=False, num_workers = 8)

def load_scores(path=None, csv=None):
    path_csv = path + csv
    data = pd.read_csv(path_csv)
    scores = data['score']
    print('so far so good')

def split_csv_scores(path=None, csv=None):
    path_csv = path + csv
    data = pd.read_csv(path_csv)
    genuines = data[data['probe_subject_id']==data['bio_ref_subject_id']]
    imposters = data[data['probe_subject_id']!=data['bio_ref_subject_id']]
    return genuines['score'], imposters['score']


def _abs_diff(a, b, cost):
    return abs(a - b)

def _weighted_err(far, frr, cost):
    return (cost * far) + ((1.0 - cost) * frr)

def _minimizing_threshold(negatives, positives, criterion, cost=0.5):

    if criterion not in ("absolute-difference", "weighted-error"):
        raise ValueError("Uknown criterion")

    def criterium(a, b, c):
        if criterion == "absolute-difference":
            return _abs_diff(a, b, c)
        else:
            return _weighted_err(a, b, c)

    if not len(negatives) or not len(positives):
        raise RuntimeError(
            "Cannot compute threshold when no positives or "
            "no negatives are provided"
        )

    # iterates over all possible far and frr points and compute the predicate
    # for each possible threshold...
    min_predicate = 1e8
    min_threshold = 1e8
    current_predicate = 1e8

    # we start with the extreme values for far and frr
    far = 1.0
    frr = 0.0

    # the decrease/increase for far/frr when moving one negative/positive
    max_neg = len(negatives)
    far_decrease = 1.0 / max_neg
    max_pos = len(positives)
    frr_increase = 1.0 / max_pos

    # we start with the threshold based on the minimum score
    # iterates until one of these goes bananas
    pos_it = 0
    neg_it = 0
    current_threshold = min(negatives[neg_it], positives[pos_it])

    # continues until one of the two iterators reaches the end of the list
    while pos_it < max_pos and neg_it < max_neg:

        # compute predicate
        current_predicate = criterium(far, frr, cost)
        if current_predicate <= min_predicate:
            min_predicate = current_predicate
            min_threshold = current_threshold
        if positives[pos_it] >= negatives[neg_it]:
            # compute current threshold
            current_threshold = negatives[neg_it]
            neg_it += 1
            far -= far_decrease
        else:  # pos_val <= neg_val
            # compute current threshold
            current_threshold = positives[pos_it]
            pos_it += 1
            frr += frr_increase
        # skip until next "different" value, which case we "gain" 1 unit on
        # the "FAR" value, since we will be accepting that negative as a
        # true negative, and not as a false positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while neg_it < max_neg and current_threshold == negatives[neg_it]:
            neg_it += 1
            far -= far_decrease
        # skip until next "different" value, which case we "loose" 1 unit
        # on the "FRR" value, since we will be accepting that positive as a
        # false negative, and not as a true positive anymore.  we continue
        # to do so for as long as the current threshold matches the current
        # iterator.
        while pos_it < max_pos and current_threshold == positives[pos_it]:
            pos_it += 1
            frr += frr_increase
        # computes a new threshold based on the center between last and current
        # score, if we are **not** already at the end of the score lists
        if neg_it < max_neg or pos_it < max_pos:
            if neg_it < max_neg and pos_it < max_pos:
                current_threshold += min(negatives[neg_it], positives[pos_it])
            elif neg_it < max_neg:
                current_threshold += negatives[neg_it]
            else:
                current_threshold += positives[pos_it]
            current_threshold /= 2

    # now, we have reached the end of one list (usually the negatives) so,
    # finally compute predicate for the last time
    current_predicate = criterium(far, frr, cost)
    if current_predicate < min_predicate:
        min_predicate = current_predicate
        min_threshold = current_threshold

    # now we just double check choosing the threshold higher than all scores
    # will not improve the min_predicate
    if neg_it < max_neg or pos_it < max_pos:
        last_threshold = current_threshold
        if neg_it < max_neg:
            last_threshold = numpy.nextafter(negatives[-1], negatives[-1] + 1)
        elif pos_it < max_pos:
            last_threshold = numpy.nextafter(positives[-1], positives[-1] + 1)
        current_predicate = criterium(0.0, 1.0, cost)
        if current_predicate < min_predicate:
            min_predicate = current_predicate
            min_threshold = last_threshold

    # return the best threshold found
    return min_threshold

def eer_threshold(negatives, positives, is_sorted=False):

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)

    return _minimizing_threshold(neg, pos, "absolute-difference")


def min_weighted_error_rate_threshold(negatives, positives, cost, is_sorted=False):

    # if not pre-sorted, copies and sorts
    neg = negatives if is_sorted else numpy.sort(negatives)
    pos = positives if is_sorted else numpy.sort(positives)
    if cost > 1.0:
        cost = 1.0
    elif cost < 0.0:
        cost = 0.0

    return _minimizing_threshold(neg, pos, "weighted-error", cost)

def min_hter_threshold(negatives, positives, is_sorted=False):
    return min_weighted_error_rate_threshold(negatives, positives, 0.5, is_sorted)

def far_threshold(negatives, positives, far_value=0.001, is_sorted=False):

    if far_value < 0.0 or far_value > 1.0:
        raise RuntimeError("`far_value' must be in the interval [0.,1.]")

    if len(negatives) < 2:
        raise RuntimeError("the number of negative scores must be at least 2")

    epsilon = numpy.finfo(numpy.float64).eps
    # if not pre-sorted, copies and sorts
    scores = negatives if is_sorted else numpy.sort(negatives) # 看下这个是从小到大排列还是从大到小，从小到达

    # handles special case of far == 1 without any iterating
    if far_value >= (1 - epsilon):
        return numpy.nextafter(scores[0], scores[0] - 1)

    # Reverse negatives so the end is the start. This way the code below will
    # be very similar to the implementation in the frr_threshold function. The
    # implementations are not exactly the same though.
    scores = numpy.flip(scores)

    # Move towards the end of array changing the threshold until we cross the
    # desired FAR value. Starting with a threshold that corresponds to FAR ==
    # 0.
    total_count = len(scores)
    current_position = 0

    # since the comparison is `if score >= threshold then accept as genuine`,
    # we can choose the largest score value + eps as the threshold so that we
    # can get for 0% FAR.
    valid_threshold = numpy.nextafter(
        scores[current_position], scores[current_position] + 1
    )
    current_threshold = 0.0

    while current_position < total_count:

        current_threshold = scores[current_position]
        # keep iterating if values are repeated
        while (
            current_position < (total_count - 1)
            and scores[current_position + 1] == current_threshold
        ):
            current_position += 1
        # All the scores up to the current position and including the current
        # position will be accepted falsely.
        future_far = (current_position + 1) / total_count
        if future_far > far_value:
            break
        valid_threshold = current_threshold
        current_position += 1

    return valid_threshold

def get_thres(criter, neg, pos, far=None):
    """Get threshold for the given positive/negatives scores and criterion

    Parameters
    ----------
    criter :
        Criterion (`eer` or `hter` or `far`)
    neg : :py:class:`numpy.ndarray`:
        array of negative scores
        pos : :py:class:`numpy.ndarray`::
        array of positive scores

    Returns
    -------
    :py:obj:`float`
        threshold
    """
    if criter == "eer":
        return eer_threshold(neg, pos)
    elif criter == "min-hter":
        return min_hter_threshold(neg, pos)
    elif criter == "far":
        if far is None:
            raise ValueError(
                "FAR value must be provided through "
            )
        return far_threshold(neg, pos, far)
    else:
        raise ValueError("Incorrect criterion: ``%s``" % criter)


def farfrr(negatives, positives, threshold):

    if numpy.isnan(threshold):
        print("Error: Cannot compute FPR (FAR) or FNR (FRR) with NaN threshold")
        return (1.0, 1.0)

    if not len(negatives):
        raise RuntimeError(
            "Cannot compute FPR (FAR) when no negatives are given"
        )

    if not len(positives):
        raise RuntimeError(
            "Cannot compute FNR (FRR) when no positives are given"
        )

    return (negatives >= threshold).sum() / len(negatives), (positives < threshold).sum() / len(positives)

def Metrics(path=None, csv=None, thr=None, criter=None, far=None):
    gen, imp = split_csv_scores(path=path, csv=csv)
    if thr is None:
        thr = get_thres(criter=criter, neg=imp, pos=gen, far=far)
    fmr, fnmr = farfrr(negatives=imp, positives=gen, threshold=thr)
    hter = (fmr + fnmr) /2.0
    if criter == 'eer':
        eer = hter
    else:
        eer = None
    if criter == 'far':
        tar_at_far = 1 - fnmr
    else:
        tar_at_far = None
    print('number of genuines: ', len(gen), '\n',
          'number of imposters: ', len(imp), '\n',
          # 'Threhold: ', thr, '\n',
          'FMR/FAR(%): ', fmr*100, '\n',
          'FNMR/FRR(%): ', fnmr*100, '\n',
          'HTER(%): ', hter*100, '\n',
          'EER(%): ', eer*100 if eer is not None else eer, '\n',
          'TAR@FAR(%): ', tar_at_far*100 if tar_at_far is not None else tar_at_far)
    print('-'*100)
    return len(gen), len(imp), thr, fmr*100, fnmr*100, hter*100, eer*100 if eer is not None else eer, tar_at_far*100 if tar_at_far is not None else tar_at_far


def save_csv(csv_data, csv_path, csv_name, csv_title):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    path_csv = csv_path + csv_name
    csv = pd.DataFrame(columns=csv_title, data=csv_data)
    csv.to_csv(path_csv, index=False)

def get_model(model_name=None):
    if model_name == 'fdt_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
        use_ba=True, batch_train=BA,
        use_lffn=True,
        use_mlppatch=True,
        use_peg=True)
        use_ba = True
    elif model_name == 'fdt_wo_ba_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
        use_ba=False, batch_train=BA,
        use_lffn=True,
        use_mlppatch=True,
        use_peg=True)
        use_ba = False
    elif model_name == 'fdt_wo_lf_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
        use_ba=True, batch_train=BA,
        use_lffn=False,
        use_mlppatch=True,
        use_peg=True)
        use_ba = True
    elif model_name == 'fdt_wo_mp_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
        use_ba=True, batch_train=BA,
        use_lffn=True,
        use_mlppatch=False,
        use_peg=True)
        use_ba = True
    elif model_name == 'fdt_wo_pe_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(144/150, 121/145, 100/122, 81/101, 64/82, 49/65), keep_rate_sv=(100/100, 81/100, 64/82, 49/65), fuse_token=True, use_fa=False,
        use_ba=True, batch_train=BA,
        use_lffn=True,
        use_mlppatch=True,
        use_peg=False)
        use_ba = True
    elif model_name == 'fdt_wo_dr_mffv':
        model = FDT(img_size=(240, 320), patch_size=(24, 32), in_chans=1, num_classes=1000, embed_dim=128,act_layer=nn.LeakyReLU, qkv_bias=True, attn_drop_rate=0., norm_layer=nn.LayerNorm, num_heads=4, mlp_ratio=8.,  drop_rate=0., depth_fv=6, depth_sv=4, keep_rate_fv=(300/300, 300/300, 300/300, 300/300, 300/300, 300/300), keep_rate_sv=(100/100, 100/100, 100/100, 100/100), fuse_token=True, use_fa=False, 
        use_ba=True, batch_train=BA, 
        use_lffn=True, 
        use_mlppatch=True,
        use_peg=True)
        use_ba = True
    elif model_name == 'mvcnn_ori_mffv_lgpu':
        model = MVCNN_ori()
        use_ba = False
    elif model_name == 'mvcnn_imp_mffv':
        model = MVCNN_imp()
        use_ba = False
    elif model_name == 'mvt_ori_mffv':
        model = MVT_ori()
        use_ba = False
    elif model_name == 'mvt_imp_mffv':
        model = MVT_imp()
        use_ba = False
    else:
        print('please provide the model name!!!')
    return model


def inference(model=None, path_save=None, prefix=None, pt=None):
    title = ['probe_subject_id', 'bio_ref_subject_id', 'score']

    if pt == 'bal':
        dev_loader = bal_dev_loader
        test_loader = bal_test_loader
    elif pt == 'nom':
        dev_loader = nom_dev_loader
        test_loader = nom_test_loader
    else:
        print('please provide the protocol!!!')

    dev_eer, devThresh, minV, frr_list, far_list, scores, probe, bio  = dev_epoch(dev_loader, model)
    print('dev eer: ', dev_eer)
    csv_data = [probe.tolist(), bio.tolist(), scores.tolist()]
    csv_data = [[row[i] for row in csv_data] for i in range(len(csv_data[0]))]
    save_csv(csv_data=csv_data, csv_path=path_save, csv_name=prefix+'dev_scores.csv', csv_title=title)

    _, _, _, _, _, scores, probe, bio  = dev_epoch(test_loader, model)
    csv_data = [probe.tolist(), bio.tolist(), scores.tolist()]
    csv_data = [[row[i] for row in csv_data] for i in range(len(csv_data[0]))]
    save_csv(csv_data=csv_data, csv_path=path_save, csv_name=prefix+'test_scores.csv', csv_title=title)

def all_inference(model_list=None, path_save=None, ds_type=None):
    for m in model_list:
        model = get_model(model_name=m)
        params_path = 'rs/trained_models/' + '/' + m + '/best/best.ckpt'
        if torch.cuda.is_available():
            model.cuda()
            model.load_state_dict(torch.load(params_path))
        else:
            model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
        for pt in ['bal', 'nom']:
            prefix = m + '_' + ds_type + '_' + pt + '_' 
            inference(model=model, path_save=path_save, prefix=prefix, pt=pt)


def all_metrics(path_save=None, model_list=None, ds_type=None):
    title = ['model', 'eer_dev', 'fmr_dev', 'fnmr_dev', 'hter_dev', 'fmr_test', 'fnmr_test', 'hter_test',
             '/', 'fmr_dev', 'fnmr_dev', 'hter_dev', 'fmr_test', 'fnmr_test', 'hter_test',
             '/', 'tar@far=0.01', 'fmr_dev', 'fnmr_dev', 'hter_dev', 'fmr_test', 'fnmr_test', 'hter_test',
             '/', 'tar@far=0.001', 'fmr_dev', 'fnmr_dev', 'hter_dev', 'fmr_test', 'fnmr_test', 'hter_test',]
    for pt in ['bal', 'nom']:
        table = []
        csv_name = pt + '_metrics_table.csv'
        for m in model_list:
            csv_dev = m + '_' + pt + '_dev_scores.csv'
            csv_test = m + '_' + pt + '_test_scores.csv'
            metrics = []
            metrics.append(m)
            # when using the eer criteria
            gen_dev, imp_dev, thr_dev, fmr_dev, fnmr_dev, hter_dev, eer_dev, tar_at_far_dev = Metrics(path=path_save, csv=csv_dev, thr= None, criter='eer', far=None)
            _, _, _, fmr_test, fnmr_test, hter_test, _, tar_at_far_test = Metrics(path=path_save, csv=csv_test, thr= thr_dev, criter=None, far=None)
            metrics.append(eer_dev); metrics.append(fmr_dev); metrics.append(fnmr_dev); metrics.append(hter_dev); metrics.append(fmr_test); metrics.append(fnmr_test); metrics.append(hter_test)
            # when using the min-hter criteria
            metrics.append('/')
            gen_dev, imp_dev, thr_dev, fmr_dev, fnmr_dev, hter_dev, eer_dev, tar_at_far_dev = Metrics(path=path_save, csv=csv_dev, thr= None, criter='min-hter', far=None)
            _, _, _, fmr_test, fnmr_test, hter_test, _, tar_at_far_test = Metrics(path=path_save, csv=csv_test, thr= thr_dev, criter=None, far=None)
            metrics.append(fmr_dev); metrics.append(fnmr_dev); metrics.append(hter_dev); metrics.append(fmr_test); metrics.append(fnmr_test); metrics.append(hter_test)
            # when using the tar=0.01 criteria
            metrics.append('/')
            gen_dev, imp_dev, thr_dev, fmr_dev, fnmr_dev, hter_dev, eer_dev, tar_at_far_dev = Metrics(path=path_save, csv=csv_dev, thr= None, criter='far', far=0.01)
            _, _, _, fmr_test, fnmr_test, hter_test, _, tar_at_far_test = Metrics(path=path_save, csv=csv_test, thr= thr_dev, criter=None, far=None)
            metrics.append(tar_at_far_dev); metrics.append(fmr_dev); metrics.append(fnmr_dev); metrics.append(hter_dev); metrics.append(fmr_test); metrics.append(fnmr_test); metrics.append(hter_test)
            # when using the tar=0.001 criteria
            metrics.append('/')
            gen_dev, imp_dev, thr_dev, fmr_dev, fnmr_dev, hter_dev, eer_dev, tar_at_far_dev = Metrics(path=path_save, csv=csv_dev, thr= None, criter='far', far=0.001)
            _, _, _, fmr_test, fnmr_test, hter_test, _, tar_at_far_test = Metrics(path=path_save, csv=csv_test, thr= thr_dev, criter=None, far=None)
            metrics.append(tar_at_far_dev); metrics.append(fmr_dev); metrics.append(fnmr_dev); metrics.append(hter_dev); metrics.append(fmr_test); metrics.append(fnmr_test); metrics.append(hter_test)
            table.append(metrics)
        save_csv(csv_data=table, csv_path=path_save, csv_name=csv_name, csv_title=title)




if __name__ == '__main__':
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    # first step, run the inference and get the scores file 
    all_inference(model_list=args.model, ds_type=args.ds_type, path_save=args.results) # rs means generate the scores' csv file in the folder name 'rs'
    # second step, calculate the metrics using different criteria, and save the metrics in the csv file
    # all_metrics(path_save=args.results, model_list=args.model, ds_type=args.ds_type)
