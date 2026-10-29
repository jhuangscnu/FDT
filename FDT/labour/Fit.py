import os
import time
from labour.EpochOperation import train_epoch, dev_epoch
import torch
import pandas as pd
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(csv_data, csv_path, csv_name, title):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    path_csv = csv_path + csv_name
    csv = pd.DataFrame(columns=title, data=csv_data)
    csv.to_csv(path_csv, index=False)

def fit(args, train_loader, dev_loader, last_epoch,  model, loss_fn1, loss_fn2, optimizer, scheduler, n_epochs, model_save_path=None, print_label=None, use_ba=False, beta=None):
    # some initializations
    title = ['probe_subject_id', 'bio_ref_subject_id', 'score']
    best_eer = 1.0
    best_model_path = os.path.join(model_save_path, 'best/')
    mkdir(best_model_path)
    bestThresh = 1
    dev_eer = 1
    minV = 1
    frr_list = []
    far_list = []
    log_path = os.path.join(model_save_path, 'log.txt')
    best_model_log_path = os.path.join(best_model_path, 'best_log.txt')
    continue_model_path = os.path.join(model_save_path, 'continue_model')
    mkdir(continue_model_path)
    loss_data = []
    for epoch in range(n_epochs):
        epoch = epoch + last_epoch
        # print('training: ', args.model, 'on ', args.dataset)
        start = time.time()
        # train
        mean_loss, mean_loss_1, mean_loss_2 = train_epoch(use_ba, train_loader, model, loss_fn1, loss_fn2, optimizer, beta)# 
        scheduler.step()
        message = 'Epoch: {}/{}. Mean loss: {:.8f}. Mean loss 1: {:.8f}. Mean loss 2: {:.8f}'.format(epoch + 1, n_epochs+last_epoch, mean_loss, mean_loss_1, mean_loss_2)
        loss_data.append([mean_loss, mean_loss_1, mean_loss_2])
        # authentication
        if epoch <= 1999: # skip the first N epoch, just for quickly training
            pass
        else:
            dev_eer, bestThresh, minV, frr_list, far_list, scores, probe, bio  = dev_epoch(dev_loader, model)
        message += '\nEpoch: {}/{}. dev eer: {:.8f}.'.format(epoch + 1, n_epochs+last_epoch, dev_eer)
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_model_name = os.path.join(best_model_path, 'best.ckpt')
            torch.save(model.state_dict(), best_model_name)
            best_model_log = '\nEpoch: {}/{} \ndev set eer:{:.8f}. \nbestThresh:{:.8f}  \nfrr list:{} \nfar list:{}.'.format(epoch+1, n_epochs, best_eer, bestThresh, frr_list, far_list)
            log_file = open(best_model_log_path, 'a')
            log_file.write(best_model_log)
            log_file.write('\n')
            log_file.close()
            csv_data = [probe.tolist(), bio.tolist(), scores.tolist()]
            csv_data = [[row[i] for row in csv_data] for i in range(len(csv_data[0]))]
            save_csv(csv_data=csv_data, csv_path=best_model_path, csv_name='dev_scores.csv', title=title)

        message+= '\nBest eer: {:.8f}'.format(best_eer)
        print(message)
        if (epoch+1)%500 == 0:
            continue_model_name = os.path.join(continue_model_path, str(epoch+1)+'.ckpt')
            torch.save(model.state_dict(), continue_model_name)
        log_file = open(log_path, 'a')
        log_file.write(message)
        log_file.write('\n')
        log_file.close()
        Fit_epoch_time = time.time() - start
        print("Fit_epoch_time: ", Fit_epoch_time)
        print('='*50)
    save_csv(csv_data=loss_data, csv_path=model_save_path, csv_name='loss.csv', title=['loss', 'loss1', 'loss2'])
