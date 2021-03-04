# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
#from torch.autograd import Variable
import logging

class OptScheduler(): ### Adam optimizer with scheduler
  def __init__(self, optimizer, size, scale, warmup, step):
    super(OptScheduler, self).__init__()
    self.optimizer = optimizer  #Adam optimizer
    self.warmup = warmup
    self.scale = scale
    self.size = size
    self._step = step           #initial step
    self._rate = 0.
    
  def lrate(self, step):
    return self.scale * (self.size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

  def step(self):
    self._step += 1                       # increments step
    self._rate =  self.lrate(self._step)  # update lrate given step
    for p in self.optimizer.param_groups: # set new lrate in optimizer
      p['lr'] = self._rate                
    self.optimizer.step()                 # parameters update (fwd) based on gradients and lrate

class LabelSmoothing_NLL(torch.nn.Module):
  def __init__(self, nclasses, pad_idx, smoothing=0.0):
    super(LabelSmoothing_NLL, self).__init__()
    self.pad_idx = pad_idx
    self.nclasses = nclasses #size of tgt vocab
    self.smoothing = smoothing

  def forward(self, pred, gold):
    #pred is [bs,lt,Vt] #logits
    #gold is [bs,lt] #references
    pred = pred.contiguous().view(-1,pred.size(2)) #[bs*lt, Vt]
    gold = gold.contiguous().view(-1) #[bs*lt]

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (self.nclasses - 1)
    log_prb = F.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(self.pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()
    return loss


class LabelSmoothing_KLDiv(torch.nn.Module):
  def __init__(self, nclasses, pad_idx, smoothing=0.0):
    super(LabelSmoothing_KLDiv, self).__init__()
    assert nclasses > 0
    assert 0.0 < smoothing <= 1.0
    assert 0 <= pad_idx <= nclasses
    self.confidence = 1.0 - smoothing
    self.pad_idx = pad_idx

    smoothing_value = smoothing / (nclasses - 2) #smoothing value
    one_hot = torch.full((nclasses,), smoothing_value) #[Vt, 1] filled with smoothing_value
    one_hot[pad_idx] = 0.0
    self.register_buffer('one_hot', one_hot.unsqueeze(0))

  def forward(self, pred, gold):
    pred = F.log_softmax(pred, dim=-1)
    #pred is [bs, lt, Vt] (after log_softmax)
    #gold is [bs, lt]
    pred = pred.contiguous().view(-1,pred.size(2)) #[bs*lt, Vt]
    gold = gold.contiguous().view(-1) #[bs*lt]

    smooth_gold_prob = self.one_hot.repeat(gold.size(0), 1) #[bs*lt, Vt]
    smooth_gold_prob.scatter_(1, gold.unsqueeze(1), self.confidence) #replaces smooth_value by confidence in gold tokens
    smooth_gold_prob.masked_fill_((gold == self.pad_idx).unsqueeze(1), 0.0) #replaces smooth_value by 0.0 in padded tokens

    loss = F.kl_div(pred, smooth_gold_prob, reduction='sum')
    return loss




