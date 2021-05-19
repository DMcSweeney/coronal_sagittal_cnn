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
