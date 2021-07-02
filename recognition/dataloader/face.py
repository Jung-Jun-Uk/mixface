import sys
import yaml
import os
import io

import torch
from torch.utils import data
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as Ap
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import re
import random
import pickle

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from dataloader.utils import create_face_dataset

class FaceDatasets(data.Dataset):
    def __init__(self, config='data/kface.yaml', mode='train'):
        assert mode in ['train', 'test']
        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.double = cfg['double']
        self.mode = mode
        self.data_path = cfg['ms1m_path'] 
        img_size = cfg['img_size']
        preprocessed_file = cfg['preprocessed_file']
        min_image = cfg['min_img']
        if not os.path.isfile(preprocessed_file):
            info_dict = create_face_dataset(self.data_path)
            with open(preprocessed_file, 'wb') as f:
                pickle.dump(info_dict, f)
        with open(preprocessed_file, 'rb') as f:
            self.info_dict = pickle.load(f)
        
        info = []
        label = 0
        for img_path_lst in self.info_dict.values():
            if len(img_path_lst) < min_image:
                continue
            for img_path in img_path_lst:
                info.append({'img_path' : img_path, 'label' : label})
            label += 1
        self.information = info
        self.num_classes = label

        print("\nCreat kface {} dataset".format(mode))
        print("Number of images        : ", len(self.information))
        print("Number of classes       : ", self.num_classes)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size,img_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
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
        img_path = info['img_path']
        label = info['label']
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
            data, hfdata = self.transform(img), self.transform(TF.hflip(img))
            return data, hfdata, label, info['img_path']

    def __len__(self):
        return len(self.information)


class BinDatasets(data.Dataset):
    def __init__(self, bin_path, config='/data/kface.yaml'):
        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        img_size = cfg['img_size']
        bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
        self.information = list()
        for i in range(0, len(issame_list)*2, 2):
            data1, data2, label = bins[i], bins[i+1], issame_list[int(i/2)]
            self.information.append({'data1' : data1, 'data2' : data2, 'label' : label})    
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        info = self.information[index]
        data1 = info['data1']
        data2 = info['data2']
        label = info['label']
        
        data1 = Image.open(io.BytesIO(data1))
        data2 = Image.open(io.BytesIO(data2))

        data1, hfdata1 = self.transform(data1), self.transform(TF.hflip(data1))
        data2, hfdata2 = self.transform(data2), self.transform(TF.hflip(data2))

        return data1, hfdata1, data2, hfdata2, label
    
    def __len__(self):
        return len(self.information)


class Face(object):
    def __init__(self, config, batch_size, test_batch_size, cuda, workers, is_training, rank):
        if rank in [-1, 0]:
            print("Face processing .. ")

        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        bin_path = cfg['bin_path']

        pin_memory = True if cuda else False

        if is_training:
            train_dataset = FaceDatasets(config=config, mode='train')

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
            trainloader = torch.utils.data.DataLoader(
                                train_dataset, 
                                batch_size=batch_size, 
                                shuffle=(train_sampler is None),
                                num_workers=workers, 
                                pin_memory=pin_memory, 
                                sampler=train_sampler)

            self.trainloader = trainloader

            self.num_classes = train_dataset.num_classes
            self.num_training_images = len(train_dataset.information)


        test_dataset = BinDatasets(bin_path=bin_path, config=config)        
        testloader = torch.utils.data.DataLoader(
                        test_dataset,                        
                        batch_size=test_batch_size, 
                        shuffle=False,
                        num_workers=workers, 
                        pin_memory=pin_memory,
                        )
                
        self.testloader = testloader
        
        if is_training and rank in [-1, 0]:
            print("len trainloader", len(self.trainloader))
            print("len testloader", len(self.testloader))

if __name__ == "__main__":
    Face(config='data/kface.yaml', batch_size=512, test_batch_size=512, cuda=True, workers=4, rank=-1)