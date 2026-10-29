from tqdm import tqdm
import os
import time
import torch
from caculate.ResultsCaculate import batch_cos_distance, embedding_calc_eer_frr_far
import torch.nn.functional as F
import numpy
from sklearn.metrics.pairwise import cosine_similarity


def train_epoch(use_ba, train_loader, model, loss_fn1, loss_fn2, optimizer1, beta):
    # some initializations
    model.train()
    start = time.time() # 开始时间
    batch_loss  = 0
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    # authentication
    for batch_idx, (img1, img2, img3, target) in enumerate(train_loader):
        data = [img1, img2, img3]
        if torch.cuda.is_available():
            data = [d.cuda() for d in data]
        if torch.cuda.is_available():
            target = target.cuda()
        optimizer1.zero_grad()
        # inference
        c, f = model(data)
        # for uasing batch attention
        if use_ba & model.training:
            target = torch.cat([target, target], dim=0)
        # caculate loss
        loss_1 = loss_fn1(c, target)
        loss_2 = loss_fn2(f, target)
        loss = loss_1 + loss_2 * beta
        total_loss_1 += loss_1.item()
        total_loss_2 += loss_2.item()
        total_loss += loss.item()
        # BP
        loss.backward()
        # update lr
        optimizer1.step()
        # print message
        message = 'Train: [{}/{} ({:.0f}%)]\t  Loss: {:.8f}'.format(batch_idx * len(data[0]), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss) 
        print(message)
    # caculate the mean losses and return
    mean_loss = total_loss / (batch_idx + 1)
    mean_loss_1 = total_loss_1 / (batch_idx + 1)
    mean_loss_2 = total_loss_2 * beta / (batch_idx + 1)
    train_epoch_time = time.time() - start
    print("train_epoch_time: ", train_epoch_time)
    return mean_loss, mean_loss_1, mean_loss_2


def dev_epoch(dev_loader, model):
    model.eval()
    start = time.time()
    cos_distances = []
    labels = []
    probe_list = []
    bio_list = []
    with torch.no_grad():
        for (img1_1, img1_2, img1_3, img2_1, img2_2, img2_3, label, probe_subject_id, bio_ref_subject_id) in tqdm(dev_loader):
            if torch.cuda.is_available():
                img1 = [img1_1.cuda(), img1_2.cuda(), img1_3.cuda()]
                img2 = [img2_1.cuda(), img2_2.cuda(), img2_3.cuda()]
                label = label.cuda()
            else:
                img1 = [img1_1, img1_2, img1_3]
                img2 = [img2_1, img2_2, img2_3]
                label = label
            # get features from authentication pair
            _, feature1 = model(img1)
            _, feature2 = model(img2)
            # cosin distance
            current_batch_cos_distance = F.cosine_similarity(feature1, feature2)
            cos_distances.append(current_batch_cos_distance)
            labels.append(label)
            probe_list.append(probe_subject_id)
            bio_list.append(bio_ref_subject_id)
        # get socres, probe, and list
        score_list = torch.cat(cos_distances)
        probe_list = torch.cat(probe_list)
        bio_list = torch.cat(bio_list)
        label = torch.cat(labels)
        eer, bestThresh, minV, frr_list, far_list = embedding_calc_eer_frr_far(score_list, label)
    test_epoch_time = time.time() - start
    print("Dev epoch time: ", test_epoch_time)
    return eer, bestThresh, minV, frr_list, far_list, score_list, probe_list, bio_list


