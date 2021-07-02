import os
import sys
from pathlib import Path
import yaml
import argparse

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from utils.general import select_device, Logger
from dataloader.load import load_datasets
from dataloader.utils import read_test_pair_dataset
from models.build import build_models
from models.head import MixFace

def extraction(model, testloader, device):
    model.eval()
    extract_info = dict()
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            data, hfdata, labels, img_path = batch
            data, hfdata = data.to(device), hfdata.to(device)

            df = model(data)
            hfdf = model(hfdata)
            f  = torch.cat((df, hfdf), 1).data.cpu()

            for idx in range(len(labels)):
                extract_info[img_path[idx]] = {'deepfeatures' : f[idx], 'label' : labels[idx]}
            
            if (i+1) % 50 == 0:
                print("Deep feature extracting ... {}/{}".format(i+1, len(testloader)))
            
    return extract_info


def cosine_similarity(x1, x2):
    """
    ex) x1 size [256, 512], x2 size [256, 512]
    similarity size = [256, 1]
    """
    x1 = F.normalize(x1).unsqueeze(1)
    x2 = F.normalize(x2).unsqueeze(1)
    
    x2t = torch.transpose(x2, 1, 2)
    similarity = torch.bmm(x1, x2t).squeeze()
    return similarity


def torch_cal_accuracy(y_score, y_true):
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th).long()
        acc = torch.mean((y_test == y_true).float())
        if acc > best_acc:
            best_acc = acc
            best_th = th
        if (i+1) % 10000 == 0 or (i+1) == len(y_score):
            print('Progress {}/{}'.format((i+1),len(y_score)))
    return best_acc, best_th        


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)   
    y_true = np.asarray(y_true)
    
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th).astype(int)

        acc = np.mean((y_test == y_true))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th


def verification(extract_info, test_pair_txt, device, split_batch_size=1024):
    print("\nRun verification ..\n")
    test_pair_info = read_test_pair_dataset(test_pair_txt)
    id1_deepfeatures = []
    id2_deepfeatures = []
    labels = []
    print("\ndeepfeatures appending .. \n")
    for id1, id2, label in test_pair_info:
        df1 = extract_info[id1]['deepfeatures']
        df2 = extract_info[id2]['deepfeatures']
        
        id1_deepfeatures.append(df1)
        id2_deepfeatures.append(df2)
        labels.append(int(label))
        
    print("\nDone ..\n")
    id1_deepfeatures = torch.stack(id1_deepfeatures, dim=0)
    id2_deepfeatures = torch.stack(id2_deepfeatures, dim=0)
    
    split_df1 = torch.split(id1_deepfeatures, split_batch_size)
    split_df2 = torch.split(id2_deepfeatures, split_batch_size)
    
    similarity = []
    for i, (df1, df2) in enumerate(zip(split_df1, split_df2)):
        df1 = df1.to(device)
        df2 = df2.to(device)
        sim = cosine_similarity(df1,df2)
        similarity.extend(sim.data.cpu().tolist())
        
    print("\nThe similarity calculation is done\n")
    similarity = torch.Tensor(similarity).to(device)
    labels = torch.Tensor(labels).to(device)
    acc, th = torch_cal_accuracy(similarity, labels)
    acc, th = float(acc.data.cpu()), float(th.data.cpu())
    print('cosine verification accuracy: ', acc, 'threshold: ', th)
    return acc, th

    
def kevaluation(model, testloader, device, test_pair_txt, save_deepfeatures=None, is_training=False):
    if save_deepfeatures is not None and os.path.isfile(save_deepfeatures) and not is_training:
        extract_info = torch.load(save_deepfeatures)
    else:
        extract_info = extraction(model, testloader, device)    
        if not is_training:
            torch.save(extract_info, save_deepfeatures)
    
    vacc, vth = verification(extract_info, test_pair_txt, device)
    return vacc, vth, extract_info


def evaluation(model, eval_loader, device):
    #model.train()
    model.eval()

    print('{} vaild total:'.format(len(eval_loader)))
    
    out_cosine = list()
    labels = list()

    with torch.no_grad():
        for ii, data in enumerate(eval_loader):
 
            data1, hfdata1, data2, hfdata2, label = data
            data1, hfdata1, data2, hfdata2 = \
                data1.to(device), hfdata1.to(device), \
                data2.to(device), hfdata2.to(device)
            
            df1    = model(data1)
            hfdf1   = model(hfdata1)
            df2    = model(data2)
            hfdf2   = model(hfdata2)
            
            f1  = torch.cat((df1, hfdf1), 1)
            f2  = torch.cat((df2, hfdf2), 1)

            out_cosine.extend(cosine_similarity(f1,f2).data.cpu().tolist())
            labels.extend(label.data.tolist())
        
    acc, th = cal_accuracy(out_cosine, labels)
    print(' cosine verification accuracy: ', acc, 'threshold: ', th)
    return acc, th

    
def inference(opt, device):
    save_dir = Path(opt.save_dir)
    sys.stdout = Logger(save_dir / 'test_log_.txt')
    save_deepfeatures = save_dir / (opt.wname + '_deepfeatures.pth')
    #Configure
    cuda = device.type != 'cpu'
    
    dataset = load_datasets(opt.dataset, opt.data_cfg, opt.batch_size, opt.batch_size, 
                            cuda, opt.workers, False, opt.global_rank)
    testloader = dataset.testloader
    
    model = build_models(opt.model).to(device)    
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    
    model_state_dict = ckpt['backbone'].float().state_dict()
    model.load_state_dict(model_state_dict, strict=False)

    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.dataset == 'kface':    
        kevaluation(model, testloader, device, dataset.test_pair_txt, save_deepfeatures, is_training=False)        
    elif opt.dataset == 'face':
        vacc, vth = evaluation(model, testloader, device)
    elif opt.dataset == 'merge':
        testloader1, testloader2 = testloader
        vacc, vth = evaluation(model, testloader1, device)
        kevaluation(model, testloader2, device, dataset.test_pair_txt, save_deepfeatures, is_training=False)
        

def parser():    
    parser = argparse.ArgumentParser(description='Face Test')
    parser.add_argument('--weights', type=str , default='', help='pretrained weights path')
    parser.add_argument('--wname'  , type=str , default='best', help='pretrained weights name: best or last')
    parser.add_argument('--dataset'           , default='kface', help='kface/face/merge')    
    parser.add_argument('--model'             , default='iresnet-34', help='iresnet-34')
    parser.add_argument('--head'              , default='arcface', help='e.g. arcface, sn-pair, ms-loss, mixface, etc.')
    parser.add_argument('--data_cfg', type=str, default='data/KFACE/kface.T4.yaml', help='data yaml path')

    parser.add_argument('--workers'           , type=int, default=4)
    parser.add_argument('--batch_size'        , type=int, default=512)

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='run test dir name')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parser()
    opt.global_rank = -1    
    opt.save_dir = Path(opt.project) / opt.name    
    #opt.save_dir = Path(opt.project) / opt.dataset / (opt.model + '-' + opt.head) / opt.name 
    #assert os.path.isdir(opt.save_dir), 'ERROR: --project_directory does not exist'    
    if opt.weights[-3:] != '.pt':
        opt.weights = opt.save_dir / 'weights' / (opt.wname + '.pt')     
    assert os.path.isfile(opt.weights), 'ERROR: --weight path does not exist'
        
    device = select_device(opt.device, batch_size=opt.batch_size, rank=opt.global_rank)
    inference(opt, device)