import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import math
from typing import Tuple
import collections


class CosineMarginProduct(nn.Module):
    '''
    @author: wujiyang
    @contact: wujiyang@hust.edu.cn
    @file: CosineMarginProduct.py
    @time: 2018/12/25 9:13
    @desc: additive cosine margin for cosface
    '''
    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output



class ArcMarginProduct(nn.Module):
    """
    @Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
             https://arxiv.org/abs/1801.07698
    """
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        #self.s = Parameter(torch.tensor(s))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = (cosine * self.cos_m - sine * self.sin_m) 
        phi = phi.type(cosine.type()) # cast a half tensor type for torch.cuda.amp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output = output * self.s
        return output


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class SimilarityBasedNpairLoss(nn.Module):    
    def __init__(self, s=16.):
        super(SimilarityBasedNpairLoss, self).__init__()
        self.s = s

    def forward(self, x, labels):
        x = F.normalize(x)
        sp, sn = convert_label_to_similarity(x, labels)        

        re_labels = torch.zeros_like(sp).long()

        sp = sp.unsqueeze(1)
        sn = sn.unsqueeze(0)

        one = torch.ones_like(sp)
        
        sn = one * sn

        output = torch.cat((sp, sn), dim=1)
    
        return self.s * output, re_labels


class Mixing_base2(nn.Module):
    """

    """
    def __init__(self, in_feature=128, out_feature=10575, e=1e-5, m=0.50, easy_margin=False):
        super(Mixing_base2, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.e = e
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))        
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # cos(theta)
        x = F.normalize(x)
        cosine = F.linear(x, F.normalize(self.weight))
        
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = (cosine * self.cos_m - sine * self.sin_m) 
        phi = phi.type(cosine.type()) # cast a half tensor type for torch.cuda.amp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        s1 = (1 / math.cos(self.m)) * (math.log(self.out_feature-1) + math.log(1-self.e) - math.log(self.e))
        output = output * s1

        sp, sn = convert_label_to_similarity(x, labels)        
        num_sn = len(sn)
        re_labels = torch.zeros_like(sp).long()

        sp = sp.unsqueeze(1)
        sn = sn.unsqueeze(0)

        one = torch.ones_like(sp)
        sn = one * sn

        s2 = (math.log(num_sn) + math.log(1-self.e) - math.log(self.e))
        output_pair = torch.cat((sp, sn), dim=1) * s2
        
        return output, output_pair, re_labels


class MixFace(nn.Module):
    """

    """
    def __init__(self, in_feature=128, out_feature=10575, e=1e-5, m=0.50, easy_margin=False):
        super(MixFace, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.e = e
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))        
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # cos(theta)
        x = F.normalize(x)
        cosine = F.linear(x, F.normalize(self.weight))
        
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = (cosine * self.cos_m - sine * self.sin_m) 
        phi = phi.type(cosine.type()) # cast a half tensor type for torch.cuda.amp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        s1 = (1 / math.cos(self.m)) * (math.log(self.out_feature-1) + math.log(1-self.e) - math.log(self.e))
        output = output * s1

        sp, sn = convert_label_to_similarity(x, labels)        
        num_sn = len(sn)
        re_labels = torch.zeros_like(sp).long()

        sp = sp.unsqueeze(1)
        sn = sn.unsqueeze(0)

        one = torch.ones_like(sp)
        sn = one * sn

        s2 = (math.log(num_sn) + math.log(1-self.e) - math.log(self.e))
        output_pair = torch.cat((sp, sn), dim=1) * s2
        
        return output, output_pair, re_labels           
        