import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import Tuple
import torch.distributed as dist

class ProxyLearning(nn.Module):
    def __init__(self, in_feature=128, num_classes=10575, me=0.2, se=1e-7, easy_margin=False):
        super(ProxyLearning, self).__init__()
        self.in_feature = in_feature
        self.num_classes = num_classes
    
        self.m = math.cos(math.pi / 4 - me)
        self.s = (1 / math.cos(self.m)) * (math.log(num_classes-1) + math.log(1-se) - math.log(se))

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, norm_x, label):
        # cos(theta)
        cosine = F.linear(norm_x, F.normalize(self.weight))
        
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
        proxy_sp = output[one_hot.bool()]
        output = output * self.s
        
        return output, proxy_sp


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class PairLearning(nn.Module):
    def __init__(self,  me=0.2, se=1e-7):
        super(PairLearning, self).__init__()
        self.m = math.cos(math.pi / 4 - me)
        self.se = se
        self.soft_plus = nn.Softplus()

    def forward(self, norm_x, proxy_sp, label):
        sp, sn = convert_label_to_similarity(norm_x, label)
        total_sp = torch.cat([proxy_sp, sp], dim=0)
        spmin = total_sp.min().detach()
        snmax = sn.max().detach()
        
        gamma_p = (1 / math.cos(self.m)) * (math.log(len(sn)-1) + math.log(1-self.se) - math.log(self.se))
        gamma_n = (math.log(len(sp)-1) + math.log(1-self.se) - math.log(self.se))
        gamma_nw = (1 / math.cos(self.m)) * (math.log(len(proxy_sp)-1) + math.log(1-self.se) - math.log(self.se))

        gamma_n = gamma_n * torch.ones_like(sp)
        gamma_nw = gamma_nw * torch.ones_like(proxy_sp)

        gamma_nt = torch.cat([gamma_nw, gamma_n])
        loss_sp = self.soft_plus(torch.logsumexp(gamma_p*(sn - spmin), dim=0)) 
        loss_sn = self.soft_plus(torch.logsumexp(gamma_nt*(snmax - total_sp), dim=0)) 
        
        loss = loss_sp + loss_sn

        """ proxy_gp = (1 / math.cos(self.m)) * (math.log(len(proxy_sp)-1) + math.log(1-self.se) - math.log(self.se))  
        gp = (math.log(len(sp)-1) + math.log(1-self.se) - math.log(self.se)) if len(sp) > 1 else 0
        gn = (1 / math.cos(self.m)) * (math.log(len(sn)-1) + math.log(1-self.se) - math.log(self.se))

        gp = gp * torch.ones_like(sp)
        proxy_gp = proxy_gp * torch.ones_like(proxy_sp)
        total_gp = torch.cat([proxy_gp, gp])
    
        loss_n = self.soft_plus(torch.logsumexp(gn*(sn - spmin), dim=0)) 
        loss_p = self.soft_plus(torch.logsumexp(total_gp*(snmax - total_sp), dim=0))
        
        loss = loss_n + loss_p """
        return loss


class MixFace(nn.Module):
    def __init__(self, in_feature=128, num_classes=10575, me=0.2, se=1e-7, easy_margin=False):
        super(MixFace, self).__init__()
        self.proxy = ProxyLearning(in_feature, num_classes, me, se, easy_margin)
        self.pairs = PairLearning(me, se)

    def forward(self, x, label, rank=-1):
        norm_x = F.normalize(x)
        output, proxy_sp = self.proxy(norm_x, label)
        if rank == -1:
            pair_loss = self.pairs(norm_x, proxy_sp, label)
        else:
            pair_loss = self.distributed_pairs(norm_x, proxy_sp, label)
        return output, pair_loss

    def distributed_pairs(self, norm_x, proxy_sp, label):
        g_norm_x_lst = [torch.zeros_like(norm_x) for _ in range(dist.get_world_size())]
        g_proxy_sp_lst = [torch.zeros_like(proxy_sp) for _ in range(dist.get_world_size())]
        g_label_lst = [torch.zeros_like(label) for _ in range(dist.get_world_size())]

        dist.all_gather(g_norm_x_lst, norm_x)
        dist.all_gather(g_proxy_sp_lst, proxy_sp)
        dist.all_gather(g_label_lst, label)

        norm_x = torch.cat(g_norm_x_lst, dim=0)
        proxy_sp = torch.cat(g_proxy_sp_lst, dim=0)
        label = torch.cat(g_label_lst, dim=0)
        pair_loss = self.pairs(norm_x, proxy_sp, label)
        return pair_loss
        
