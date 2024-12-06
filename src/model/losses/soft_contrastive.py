import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from einops import reduce

class SoftContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, sim=None, margin=0, max_violation=False, threshold_hetero=1.0, threshold_homo=1.0, **kwargs):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = sim

        self.max_violation = max_violation
        self.threshold_hetero = threshold_hetero
        self.threshold_homo = threshold_homo

   
    def forward(self, mm, s, m_mask, s_mask):
        # compute inter-modal triplet loss
        scores = self.sim(mm, s, m_mask, s_mask)

        scores_emb1 = self.sim(mm, mm, m_mask, m_mask)
        scores_emb2 = self.sim(s, s, s_mask, s_mask)

        # cost_intra = self._forward_once(scores_emb1) + self._forward_once(scores_emb2)
        cost_intra = 0
        # clear false negative samples
        drop_num = 0
        if self.max_violation:
            scores_emb1 = scores_emb1.detach() 
            scores_emb2 = scores_emb2.detach() 
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask).to(scores.device)
            scores_emb1 = scores_emb1 * ~I
            scores_emb2 = scores_emb2 * ~I
            mask_emb1 = scores_emb1 > self.threshold_hetero
            mask_emb2 = scores_emb2 > self.threshold_homo
            mask = (mask_emb1 | mask_emb2)
            I = Variable(mask).to(scores.device)
            scores = scores * ~I
            drop_num = I.sum()

        cost = self._forward_once(scores) + self._forward_once(scores.T)+ cost_intra

        return cost, drop_num
    
    def _forward_once(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)

        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # text retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(scores.device)

        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
        
        return cost_s.sum()
