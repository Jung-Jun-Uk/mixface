import yaml
import os
import sys
import datetime
import time

import argparse
from pathlib import Path
import shutil
from copy import deepcopy

from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.optimizer import CosineWarmupLr
from pytorch_metric_learning import miners

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import select_device, increment_path, Logger, AverageMeter, \
    print_argument_options, init_torch_seeds, is_parallel, get_latest_run

from dataloader.load import load_datasets
from models.build import build_models, HeadAndLoss
from test import kevaluation, evaluation


def main(opt, device):

    save_dir = Path(opt.save_dir)   
    # Hyperparameters
    with open(opt.data_cfg) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.head_cfg) as f:
        head_cfg = yaml.load(f, Loader=yaml.FullLoader)

    sys.stdout = Logger(save_dir / 'log_.txt', opt.resume)
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
        
    print(vars(opt))

    # Save run settings
    with open(save_dir / 'data_cfg.yaml', 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    with open(save_dir / 'head.cfg.yaml', 'w') as f:
        yaml.dump(head_cfg, f, sort_keys=False)

    # Print the config file(opt and hyp)
    if opt.global_rank in [-1, 0]:
        print_argument_options(opt, 'Config File')
        print_argument_options(data_cfg, 'Data Config File')
        print_argument_options(hyp, 'Learning hyparperameters')
        print_argument_options(head_cfg, 'Head Config File')
        
    #Configure
    cuda = device.type != 'cpu'
    init_torch_seeds(2 + opt.global_rank)

    if opt.dataset in ['face', 'merge']:
        dcfg = opt.data_cfg
    else:
        dcfg = data_cfg    

    dataset = load_datasets(opt.dataset, dcfg, opt.batch_size, opt.total_batch_size, 
                            cuda, opt.workers, opt.global_rank)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    
    model = build_models(opt.model).to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = LabelSmoothingLoss(dataset.num_classes, smoothing=0.1)
    headandloss = HeadAndLoss(criterion=criterion, head_name=opt.head, in_feature=512, num_classes=dataset.num_classes, cfg=head_cfg).to(device)

    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model_state_dict = ckpt['backbone'].float().state_dict()
        model.load_state_dict(model_state_dict, strict=False)
        if ckpt.get('headandloss') is not None:
            head_state_dict = ckpt['headandloss'].float().state_dict()
            headandloss.load_state_dict(head_state_dict, strict=False)
        
    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if cuda and opt.global_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
        
    if opt.global_rank in [-1, 0]:
        print("Creat model  : {}".format(opt.model))
        print("Creat head   : {}".format(opt.head))

    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'], momentum=hyp['momentum'])
    optimizer.add_param_group({'params': headandloss.parameters()})

    batches_per_epoch = dataset.num_training_images // opt.total_batch_size
    scheduler = CosineWarmupLr(optimizer, batches_per_epoch, opt.max_epoch, base_lr=hyp['lr'], warmup_epochs=hyp['warmup_epochs'])
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=hyp['milestones'], gamma=hyp['gamma'])
    opt.scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Resume
    best_vacc, best_vth = 0.0, 0.0
    if pretrained:
        # Optimizer
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])        
        best_vacc = ckpt['best_vacc']
        best_vth = ckpt['best_vthreshold']
        opt.start_epoch = ckpt['epoch'] + 1
        del ckpt, model_state_dict, head_state_dict

    opt.double = data_cfg['double']
    for epoch in range(opt.start_epoch, opt.max_epoch):
        if opt.global_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        if opt.global_rank in [-1, 0]:
            print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))

        #train(opt, model, headandloss, optimizer, scheduler, trainloader, epoch, device, opt.global_rank)
        train(opt, model, headandloss, optimizer, scheduler, trainloader, testloader, epoch, device, opt.global_rank)
        #scheduler.step()
        #if epoch < 18:
        #    continue

        if opt.global_rank in [-1, 0]:
            if opt.dataset == 'kface':
                vacc, vth, extract_info = kevaluation(model, testloader, device, test_pair_txt=data_cfg['test_pair_txt'], is_training=True)        
            elif opt.dataset == 'face':
                vacc, vth = evaluation(model, testloader, 'lfw', device)
            elif opt.dataset == 'merge':
                testloader1, testloader2 = testloader
                vacc1, vth1 = evaluation(model, testloader1, 'lfw', device)
                vacc2, vth2, extract_info = kevaluation(model, testloader2, device, test_pair_txt=data_cfg['test_pair_txt'], is_training=True)                
                vacc = vacc1
                vth = vth1
            
            if vacc >= best_vacc:
                best_vacc = vacc                
                best_vth  = vth
            
            # Save backbone, head
            ckpt = {'epoch' : epoch,                     
                    'best_vacc': best_vacc, 
                    'best_vthreshold': best_vth,
                    'backbone' : deepcopy(model.module if is_parallel(model) else model).eval(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
            
            if len(headandloss.state_dict().keys()) > 0: # if the parameter exists
                ckpt['headandloss'] = deepcopy(headandloss.module if is_parallel(headandloss) else headandloss).eval()
            else:
                ckpt['headandloss'] = None

            # Save last, best and delete
            if opt.dataset == 'kface':
                torch.save(ckpt, last)
                torch.save(extract_info, save_dir / 'last_deepfeatures.pth')
                if best_vacc == vacc:
                    torch.save(ckpt, best)
                    torch.save(extract_info, save_dir / 'best_deepfeatures.pth')
                del extract_info
            
            elif opt.dataset in ['face', 'merge']:
                torch.save(ckpt, last)
                if best_vacc == vacc:
                    torch.save(ckpt, best)
            del ckpt

    
def train(opt, model, headandloss, optimizer, scheduler, trainloader, testloader, epoch, device, rank):
    model.train()
    losses = AverageMeter()
    #miner = miners.MultiSimilarityMiner()
    start_time = time.time() 
    for i, batch in enumerate(trainloader):
        if opt.double:
            data1, data2, labels = batch            
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            data, labels = torch.cat((data1, data2), dim=0), torch.cat((labels, labels), dim=0)
        else:
            data, labels = batch            
            data, labels = data.to(device), labels.to(device)
            
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            deep_features = model(data)
            loss = headandloss(deep_features, labels)
            
        opt.scaler.scale(loss).backward()
        opt.scaler.step(optimizer)
        opt.scaler.update()
        scheduler.step()

        losses.update(loss.item(), labels.size(0))
        if (i+1) % opt.print_freq == 0 and rank in [-1, 0]:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, elapsed))
            
        #if (i+1) % 5000 == 0:
        #    vacc, vth = evaluation(model, testloader, 'lfw', device)
        #    model.train()
        
def parser():    
    parser = argparse.ArgumentParser(description='Face training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--dataset'           , default='kface', help='kface/face/merge')
    parser.add_argument('--model'             , default='iresnet-34', help='iresnet-34')
    parser.add_argument('--head'              , default='arcface', help='e.g. arcface, sn-pair, ms-loss, mixface, etc.')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--use_nsps'          , action='store_true', help='adds N+S Pair Similarity Loss')
    parser.add_argument('--data_cfg', type=str, default='data/kface.large.yaml', help='data yaml path')
    parser.add_argument('--hyp', type=str, default='data/kface.hyp.yaml', help='hyperparameters path')
    parser.add_argument('--head_cfg', type=str, default='models/head.kface.cfg.yaml', help='head config path')

    parser.add_argument('--workers'          , type=int, default=4)
    parser.add_argument('--batch_size'       , type=int, default=512)
    parser.add_argument('--max_epoch'        , type=int, default=20)
    parser.add_argument('--start_epoch'      , type=int, default=0)
    parser.add_argument('--eval_freq'        , default=1)
    parser.add_argument('--print_freq'       , default=50)

    #parser.add_argument('--resume', action='store_true')
    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/inner-attribute', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    args = parser.parse_args()

    return args


# data parallel script: python train.py 
# distributed parallel script:  python -m torch.distributed.launch --nproc_per_node 4 train.py
if __name__ == "__main__":    
    opt = parser()
    opt.save_dir = increment_path(Path(opt.project) / opt.dataset / (opt.model + '-' + opt.head) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    
    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else os.path.join(opt.save_dir,'weights/last.pt')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.weights, opt.resume = ckpt, True
        opt.hyp = str(Path(ckpt).parent.parent / 'hyp.yaml')
        opt.head_cfg = str(Path(ckpt).parent.parent / 'head.cfg.yaml')
        print('Resuming training from %s' % ckpt)

    #DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size, rank=opt.global_rank)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    main(opt, device)