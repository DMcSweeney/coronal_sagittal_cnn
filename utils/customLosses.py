"""
Script with modified losses 
"""
from os import stat
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch
import torch.nn.functional
import numpy as np

def js_reg(p, q):
    #~ Jensen-Shannon Divergence
    #@params:
    # *pred + target are 1D heatmaps
    assert p.shape == q.shape, 'Predicted heatmap not same shape as target'
    #* JS(P||Q) = 0.5*D(P||M) + 0.5*D(Q||M)
    #*M = 0.5*(P+Q)
    m = 0.5*(p + q)
    return 0.5*kl_reg(p, m) + 0.5*kl_reg(q, m)

def kl_reg(q, p):
    #~ Kullback-Leibler Divergence
    eps=1e-24
    #* D(P||Q) = P log P/Q 
    #* Add small constant to keep log finite
    unsummed_kl = p*((p+eps).log() - (q+eps).log())
    return unsummed_kl.sum((-2, -1))
    
    #return unsummed_kl.sum((-2, -1))


def dice_coef(y_true, y_pred, smooth=1):
    #**
    #* Dice = (2*|X & Y|)/ (|X|+ |Y|)
    #*     =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    #* ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (torch.sum(torch.square(y_true), -1) + \
        torch.sum(torch.square(y_pred), -1) + smooth)

def dice_coef_loss(y_true, y_pred, smooth=1):
    #* Defined as 'soft dice', works well for segmentation tasks with one two classes
    return 1-dice_coef(y_true, y_pred, smooth=smooth)


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return dice_coef_loss(target, pred)


class multi_class_dice(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self, pred, target, mask=None):

        batch_size, num_classes= pred.size()[:2]
        loss = 0
        dice_pred = F.softmax(pred, dim=1)
        #dice_pred = self.sigmoid(pred)
        used_channels = 0
        for idx in range(batch_size):
            for channel in range(num_classes):
                if channel == 0: continue
                if mask is not None and mask[idx, channel-1] == 1.:
                    mask = torch.where(target[idx] == channel, 1, 0)
                    loss += dice_coef_loss(mask, dice_pred[idx, channel])
                    used_channels += 1
        return loss.mean()/used_channels


class FocalLoss(nn.Module):
    def __init__(self, pos_weight=2, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, 
        reduction='mean', pos_weight=torch.tensor([self.pos_weight]).cuda())
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class edgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, pred, coords, labels):
        # ~ Loss SUM_i( |y_top_i - *y_top_i| + |y_bot_i - *y_bot_i|) i where labels[i] == 1
        #* coords = gt coords
        loss = []
        for batch in range(pred.shape[0]):
            dists = self.get_dists(coords[batch])
            pred_dists = self.get_dists(pred[batch])
            diff = torch.abs(dists-pred_dists)  
            masked_diff = diff*labels[..., None]
            loss.append(torch.sum(masked_diff)/torch.sum(labels))
        loss = torch.tensor(loss)
        return torch.sum(loss)/pred.shape[0]

    def get_dists(self, coords):
        #~ Calculate sup-inf. distance between neighbouring verts.
        arr = []
        for i, (x, y) in enumerate(coords):
            if i == 0:
                #* Don't calc top
                bot_y = coords[i+1][-1]
                bot = torch.abs(y - bot_y)
                top = 0
            elif i == coords.shape[0]-1:
                #* Don't calc bot
                top_y = coords[i-1][-1]
                top = torch.abs(y - top_y)
                bot = 0
            else:
                top_y, bot_y = coords[i-1][-1], coords[i+1][-1]
                top = torch.abs(y - top_y)
                bot = torch.abs(y - bot_y)
            arr.append([top, bot])
        return torch.tensor(arr).to(self.device)


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


