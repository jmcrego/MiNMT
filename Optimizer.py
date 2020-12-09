# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import logging

class OptScheduler(): ### Adam optimizer with scheduler
  def __init__(self, optimizer, size, factor, warmup, step):
    super(OptScheduler, self).__init__()
    self.optimizer = optimizer  #Adam optimizer
    self.warmup = warmup
    self.factor = factor
    self.size = size
    self._step = step           #initial step
    self._rate = 0.
    
  def lrate(self, step):
    return self.factor * (self.size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

  def step(self): 
    self._step += 1                       # increments step
    self._rate =  self.lrate(self._step)  # update lrate given step
    for p in self.optimizer.param_groups: # set new lrate in optimizer
      p['lr'] = self._rate                
    self.optimizer.step()                 # parameters update (fwd) based on gradients computed and lrate

class LabelSmoothing(torch.nn.Module):
  def __init__(self, nclasses, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    self.criterion = torch.nn.KLDivLoss(reduction='sum')
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.nclasses = nclasses #size of tgt vocab

  def forward(self, pred, gold):
    #pred is [bs, lt, Vt]
    #gold is [bs, lt]
    assert pred.size(0) == gold.size(0)
    assert pred.size(1) == gold.size(1)
    assert pred.size(2) == self.nclasses

    pred = pred.contiguous().view(-1, pred.size(-1)) #[bs*lt, Vt]
    gold = gold.contiguous().view(-1).long() #gold is [bs*lt]
    return F.cross_entropy(pred, gold, ignore_index=self.padding_idx, reduction='sum')


    #true_dist is the gold distribution after label smoothing
    true_dist = pred.data.clone() #[bs*lt, Vt]
    true_dist.fill_(self.smoothing / (self.nclasses - 2))
    true_dist.scatter_(1, gold.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(gold.data == self.padding_idx, as_tuple=False)
    true_dist.index_fill_(0, mask.squeeze(), 0.0)
    return self.criterion(pred, Variable(true_dist, requires_grad=False)) ### sum of loss of all words (other than <pad> in reference)


