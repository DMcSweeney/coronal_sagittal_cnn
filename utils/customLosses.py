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

class multi_class_dice(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self, pred, target):
        num_classes = pred.size()[1]
        loss = 0
        #dice_pred = F.softmax(pred, dim=1)
        dice_pred = self.sigmoid(pred)
        for channel in range(num_classes):
            mask = torch.where(target == channel, 1, 0)
            loss += dice_coef_loss(mask, dice_pred[:, channel])
        return loss.mean()/num_classes


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean', apply_sigmoid=False):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        self.apply_sigmoid=apply_sigmoid

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = self.sigmoid(input)
        # ce_loss = F.cross_entropy(
        #     input, target, reduction=self.reduction, weight=self.weight)
        ce_loss = F.binary_cross_entropy(input, target, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

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
