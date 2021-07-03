import yaml

#from .backbone import Backbone
from .iresnet import iresnet
from .head import ArcMarginProduct, CosineMarginProduct, SimilarityBasedNpairLoss, MixFace
from pytorch_metric_learning import miners, losses
import torch
from torch import nn
import torch.nn.functional as F


def build_models(model_name, nodrop=False):
    model_args = model_name.split('-')    
    
    if model_args[0] == 'iresnet':
        _, net_depth = model_args
        model = iresnet(num_layers=int(net_depth), drop_lastfc=not nodrop)
    return model

    
class HeadAndLoss(nn.Module):
    def __init__(self, criterion, head_name, in_feature, num_classes, cfg='models/head.cfg.yaml'):
        super(HeadAndLoss, self).__init__()
        
        self.criterion = criterion
        self.head_name = head_name
        
        # Head config file
        if isinstance(cfg, str):
            with open(cfg) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        try:
            opt = cfg[head_name]
        except:
            opt = None
        print(head_name, " config")
        print(opt)

        self.miner = None        
        # Classification Loss
        if head_name == 'arcface':
            self.head = ArcMarginProduct(in_feature=in_feature, out_feature=num_classes, s=opt['s'], m=opt['m'], easy_margin=opt['easy_margin'])
        elif head_name == 'cosface':
            self.head = CosineMarginProduct(in_feature=in_feature, out_feature=num_classes, s=opt['s'], m=opt['m'])

        # Metric Loss
        elif head_name == 'n-pair-loss':
            self.head = losses.NPairsLoss()
        elif head_name == 'sn-pair-loss':
            self.head = SimilarityBasedNpairLoss(s=opt['s'])        
        elif head_name == 'ms-loss':
            if opt['use_miner']:
                self.miner = miners.MultiSimilarityMiner()
            self.head = losses.MultiSimilarityLoss()        

        # MixFace
        elif head_name == 'mixface':
            self.head = MixFace(in_feature=in_feature, out_feature=num_classes, e=float(opt['e']), m=opt['m'], easy_margin=opt['easy_margin'])
        
    def forward(self, deep_features, labels, rank=-1):
        if self.head_name in ['arcface', 'cosface']:
            outputs = self.head(deep_features, labels)
            loss = self.criterion(outputs, labels)            
        elif self.head_name in ['sn-pair-loss']:
            outputs, re_labels = self.head(deep_features, labels)
            loss = self.criterion(outputs, re_labels)            
        elif self.head_name in ['ms-loss', 'n-pair-loss']:
            if self.miner is not None:
                hard_pairs = self.miner(deep_features, labels)
                loss = self.head(deep_features, labels, hard_pairs)
            else:
                loss = self.head(deep_features, labels)   
        elif self.head_name in ['mixface']:
            outputs, outputs_pair, re_labels = self.head(deep_features, labels)
            loss = self.criterion(outputs, labels) + self.criterion(outputs_pair, re_labels)
        return loss    
        
        









