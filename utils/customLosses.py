"""
Script with modified losses 
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch
import torch.nn.functional

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
    unsummed_kl = p *((p+eps).log() - (q+eps).log())
    return unsummed_kl.sum(-1)


class FocalLoss(nn.Module):
    #* https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        print(input.shape)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else: return loss.sum()
