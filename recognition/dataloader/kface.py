import sys
import yaml
import os

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF
import albumentations as A
import albumentations.pytorch as Ap
from PIL import Image
import cv2
import re
import random


sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from dataloader.utils import kface_accessories_converter, kface_expressions_converter, \
    kface_luces_converter, kface_pose_converter, create_kface_dataset, read_test_pair_dataset

"""
    ###################################################################

    K-Face : Korean Facial Image AI Training Dataset
    url    : http://www.aihub.or.kr/aidata/73

    Directory structure : High-ID-Accessories-Lux-Emotion
    ID example          : '19062421' ... '19101513' len 400
    Accessories example : 'S001', 'S002' .. 'S006'  len 6
    Lux example         : 'L1', 'L2' .. 'L30'       len 30
    Emotion example     : 'E01', 'E02', 'E03'       len 3
    S001 - L1, every emotion folder contaions a information txt file
    (ex. bbox, facial landmark) 
    
    ###################################################################
"""


class KFaceDatasets(data.Dataset):
    def __init__(self, data_path, test_idx_path, img_size, acs, lux, eps, pose, mode='train', double=False):
        assert mode in ['train', 'test']
        
        self.mode = mode
        self.double = double
        self.data_path = data_path 
        test_idx_path = test_idx_path
        img_size = img_size

        self.information, self.num_classes = create_kface_dataset(
                                                data_path=self.data_path,                                     
                                                test_idx_path=test_idx_path,
                                                accessories=acs, 
                                                luces=lux, 
                                                expressions=eps, 
                                                poses=pose, 
                                                mode=mode)
        
        print("\nCreat kface {} dataset".format(mode))
        print("Number of images        : ", len(self.information))
        print("Number of classes       : ", self.num_classes)
        print('Aceessories : ',acs)
        print('Lux         : ',lux)
        print('Expression  : ',eps)
        print('Pose        : ',pose)

        """ info_dict = {}        
        for info in self.information:
            img_path, label = info['img_path'], info['label']            
            if info_dict.get(label) == None:
                info_dict[label] = [img_path]                
            else:                
                temp = list(info_dict[label])                
                info_dict[label] = temp + [img_path]

        self.info_dict = info_dict """

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size,img_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            """ self.transform = A.Compose([
                A.HorizontalFlip(),
                Ap.transforms.ToTensor(normalize={'mean' : [0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}),
            ]) """
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size,img_size), interpolation=Image.BICUBIC),                    
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def differ_choice(self, img_path, label):
        img_lst = self.info_dict[label]        
        if len(img_lst) == 1:
            return img_path
        others = list(set(img_lst) - set(img_path))
        return random.choice(others)

    def __getitem__(self, index):
        info = self.information[index]
        img_path = os.path.join(self.data_path, info['img_path']) 
        label    = info['label']        
        img = Image.open(img_path)
        if self.mode == 'train':        
            data = self.transform(img)            
            if self.double:
                img_path2 = self.differ_choice(img_path, label)
                img2 = Image.open(img_path2)
                data2 = self.transform(img2)
                return data, data2, label                        
            return data, label
        else:
            img = Image.open(img_path)
            data, hfdata = self.transform(img), self.transform(TF.hflip(img))
            return data, hfdata, label, info['img_path']
        
    def __len__(self):
        return len(self.information)


class KFace(object):
    def __init__(self, config, batch_size, test_batch_size, cuda, workers, rank):
        if rank in [-1, 0]:
            print("KFace processing .. ")    

        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict    
        data_path = cfg['data_path']       
           
        test_idx_path = cfg['test_idx_txt']
        self.test_pair_txt = cfg['test_pair_txt']
        img_size = cfg['img_size']

        train_acs = kface_accessories_converter(cfg['train_acs'])
        train_lux = kface_luces_converter(cfg['train_lux'])
        train_eps = kface_expressions_converter(cfg['train_eps'])
        train_pose = kface_pose_converter(cfg['train_pose'])

        test_acs = kface_accessories_converter(cfg['test_acs'])
        test_lux = kface_luces_converter(cfg['test_lux'])
        test_eps = kface_expressions_converter(cfg['test_eps'])
        test_pose = kface_pose_converter(cfg['test_pose'])

        double = cfg['double']

        train_dataset = KFaceDatasets(data_path, test_idx_path, img_size, 
                                      train_acs, train_lux, train_eps, train_pose, mode='train', double=double)
        test_dataset = KFaceDatasets(data_path, test_idx_path, img_size, 
                                     test_acs, test_lux, test_eps, test_pose, mode='test', double=False)
        
        pin_memory = True if cuda else False

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
        trainloader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            shuffle=(train_sampler is None),
                            num_workers=workers, 
                            pin_memory=pin_memory, 
                            sampler=train_sampler)
        
        
        testloader = torch.utils.data.DataLoader(
                        test_dataset,                        
                        batch_size=test_batch_size, 
                        shuffle=False,
                        num_workers=workers, 
                        pin_memory=pin_memory,
                        )
        
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.num_classes = train_dataset.num_classes
        self.num_training_images = len(train_dataset.information)
        
        if rank in [-1, 0]:
            print("len trainloader", len(self.trainloader))
            print("len testloader", len(self.testloader))
            

if __name__ == "__main__":
    pass

    
