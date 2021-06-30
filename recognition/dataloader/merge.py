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

from dataloader.utils import create_face_dataset, kface_accessories_converter, kface_expressions_converter, \
    kface_luces_converter, kface_pose_converter, create_kface_dataset, read_test_pair_dataset
from dataloader.face import BinDatasets
from dataloader.kface import KFaceDatasets


class MergeDatasets(data.Dataset): # kface + face
    def __init__(self, config='data/merge.yaml', mode='train'):
        assert mode in ['train', 'test']
        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        
        self.mode = mode        
        self.double = cfg['double']        
        self.kface_path = cfg['data_path']                 
        test_idx_path = cfg['test_idx_txt']

        acs = kface_accessories_converter(cfg['acs'])
        lux = kface_luces_converter(cfg['lux'])
        eps = kface_expressions_converter(cfg['eps'])
        pose = kface_pose_converter(cfg['pose'])
        
        self.information_kface, self.num_kface_classes = create_kface_dataset(
                                                            data_path=self.kface_path,                                     
                                                            test_idx_path=test_idx_path,
                                                            accessories=acs, 
                                                            luces=lux, 
                                                            expressions=eps, 
                                                            poses=pose, 
                                                            mode=mode)
        
        info_dict = {}        
        kface_preprocessed_file = cfg['kface_preprocessed_file']
        if not os.path.isfile(kface_preprocessed_file):
            for i, info in enumerate(self.information_kface):
                if (i+1) % 50000 == 0:
                    print(i+1, "preprocessing ...")

                img_path, label = info['img_path'], info['label']
                img_path = os.path.join(self.kface_path, img_path)            
                if info_dict.get(label) == None:
                    info_dict[label] = [img_path]                                
                else:                
                    temp = list(info_dict[label])                
                    info_dict[label] = temp + [img_path]                
            with open(kface_preprocessed_file, 'wb') as f:
                pickle.dump(info_dict, f)
        with open(kface_preprocessed_file, 'rb') as f:
            self.info_dict_kface = pickle.load(f)
            
        print('Aceessories : ',acs)
        print('Lux         : ',lux)
        print('Expression  : ',eps)
        print('Pose        : ',pose)

        self.ms1m_path = cfg['ms1m_path'] 
        img_size = cfg['img_size']
        ms1m_preprocessed_file = cfg['ms1m_preprocessed_file']
        min_image = cfg['min_img']
        if not os.path.isfile(ms1m_preprocessed_file):
            info_dict = create_face_dataset(self.ms1m_path)
            with open(ms1m_preprocessed_file, 'wb') as f:
                pickle.dump(info_dict, f)
        with open(ms1m_preprocessed_file, 'rb') as f:
            self.info_dict_face = pickle.load(f)
        
        info = []
        label = 0        
        for img_path_lst in self.info_dict_kface.values():
            for img_path in img_path_lst:
                info.append({'img_path' : img_path, 'label' : label})
            label += 1
        for img_path_lst in self.info_dict_face.values():
            if len(img_path_lst) < min_image:
                continue
            for img_path in img_path_lst:
                info.append({'img_path' : img_path, 'label' : label})
            label += 1

        self.information = info
        self.num_classes = label

        print("\nnMerge kface + face dataset {} dataset".format(mode))
        print("Number of images        : ", len(self.information))
        print("Number of classes       : ", self.num_classes)

        if mode == 'train':            
            self.transform = A.Compose([
                A.HorizontalFlip(),
                Ap.transforms.ToTensor(normalize={'mean' : [0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}),
            ])
        else:            
            self.transform = A.Compose([
                A.HorizontalFlip(),
                Ap.transforms.ToTensor(normalize={'mean' : [0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}),
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
        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            data = self.transform(image=img)
            if self.double:
                img_path2 = self.differ_choice(img_path, label)
                img2 = cv2.imread(img_path2)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                data2 = self.transform(image=img2)
                return data['image'], data2['image'], label
            return data['image'], label    
            #data = self.transform(img)
            #return data, label        
        else:
            data, hfdata = self.transform(img), self.transform(TF.hflip(img))
            return data, hfdata, label, info['img_path']

    def __len__(self):
        return len(self.information)


class Merge(object):
    def __init__(self, config, batch_size, test_batch_size, cuda, workers, rank):
        if rank in [-1, 0]:
            print("Merge Face processing .. ")

        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        bin_path = cfg['bin_path']        
        data_path = cfg['data_path']                 
        test_idx_path = cfg['test_idx_txt']
        img_size = cfg['img_size']

        acs = kface_accessories_converter(cfg['acs'])
        lux = kface_luces_converter(cfg['lux'])
        eps = kface_expressions_converter(cfg['eps'])
        pose = kface_pose_converter(cfg['pose'])

        train_dataset = MergeDatasets(config=config, mode='train')        
        test_dataset1 = BinDatasets(bin_path=bin_path, config=config)
        test_dataset2 = KFaceDatasets(data_path, test_idx_path, img_size, 
                                     acs, lux, eps, pose, mode='test', double=False)

        pin_memory = True if cuda else False

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
        trainloader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            shuffle=(train_sampler is None),
                            num_workers=workers, 
                            pin_memory=pin_memory, 
                            sampler=train_sampler)
        
        
        testloader1 = torch.utils.data.DataLoader(
                        test_dataset1,                        
                        batch_size=test_batch_size, 
                        shuffle=False,
                        num_workers=workers, 
                        pin_memory=pin_memory,
                        )
        
        testloader2 = torch.utils.data.DataLoader(
                        test_dataset2,                        
                        batch_size=test_batch_size, 
                        shuffle=False,
                        num_workers=workers, 
                        pin_memory=pin_memory,
                        )

        self.trainloader = trainloader
        self.testloader = [testloader1, testloader2]
        
        self.num_classes = train_dataset.num_classes
        self.num_training_images = len(train_dataset.information)
        
        if rank in [-1, 0]:
            print("len trainloader", len(self.trainloader))
            print("len lfw testloader", len(self.testloader[0]))
            print("len kface testloader", len(self.testloader[1]))


if __name__ == "__main__":
    pass
