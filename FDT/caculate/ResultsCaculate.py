from itertools import combinations
import numpy as np
from caculate.DistanceCaculate import cos_distance, batch_cos_distance
import torch
import math


def embedding_calc_eer_frr_far(distances, label):
    minV = 100
    bestThresh = 0
    eer = 1
    if torch.cuda.is_available():
        ones = torch.ones(label.shape).type(torch.LongTensor).cuda() 
        zeros = torch.zeros(label.shape).type(torch.LongTensor).cuda() 
    else:
        ones = torch.ones(label.shape).type(torch.LongTensor)
        zeros = torch.zeros(label.shape).type(torch.LongTensor)  
    
    frr_list = []
    far_list = []
    max_dist = torch.max(distances)
    min_dist = torch.min(distances)

    threshold_list = distances
    with torch.no_grad():
        for threshold in threshold_list:
            pred = torch.gt(distances, threshold).squeeze(-1)#.cuda()
            if torch.cuda.is_available():
                pred.cuda()
            
            if torch.cuda.is_available():
                tp = ((label == ones) & (pred==ones)).sum().type(torch.FloatTensor).cuda()    
                fn = ((label == ones) & (pred == zeros)).sum().type(torch.FloatTensor).cuda()  
                fp = ((label == zeros) & (pred == ones)).sum().type(torch.FloatTensor).cuda()  
                tn = ((label == zeros) & (pred == zeros)).sum().type(torch.FloatTensor).cuda() 
            else:
                tp = ((label == ones) & (pred==ones)).sum().type(torch.FloatTensor)
                fn = ((label == ones) & (pred == zeros)).sum().type(torch.FloatTensor)
                fp = ((label == zeros) & (pred == ones)).sum().type(torch.FloatTensor)
                tn = ((label == zeros) & (pred == zeros)).sum().type(torch.FloatTensor)
            fr = fn / (fn + tp)
            fa = fp / (fp + tn)
            frr_list.append(fr.cpu())
            far_list.append(fa.cpu())
            if abs(fr - fa) < minV:
                minV = abs(fr - fa)
                eer = (fr + fa) / 2
                bestThresh = threshold
    return eer, bestThresh, minV, np.array(frr_list).reshape(1, -1), np.array(far_list).reshape(1, -1)








