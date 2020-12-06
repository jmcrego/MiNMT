# -*- coding: utf-8 -*-

import torch
import logging

def build_AdamOptimizer(model, lr, beta1, beta2, eps): 
  return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

class OptScheduler(): ### Adam optimizer with scheduler
  def __init__(self, optimizer, size, factor, warmup, step):
    super(OptScheduler, self).__init__()
    self.optimizer = optimizer  #optimizer
    self.warmup = warmup
    self.factor = factor
    self.size = size
    self._step = step           #initial step
    self._rate = 0.0
        
  def step(self): 
    self._step += 1                                                                                                  # increments step
    self._rate = self.factor * (self.size ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup ** (-1.5))) # compute lrate given step
    for p in self.optimizer.param_groups:                                                                            # set new lrate in optimizer
      p['lr'] = self._rate
    self.optimizer.step()                                                                                            # parameters update (fwd) based on gradients computed and lrate


class LabelSmoothing(torch.nn.Module):
  def __init__(self, size, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    self.criterion = torch.nn.KLDivLoss(reduction='sum')
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size #size of tgt vocab
    self.true_dist = None

  def forward(self, pred, ref):
    #pred is [bs*lt, Vt]
    #ref is [bs*lt]
    assert pred.size(0) == ref.size(0)
    assert pred.size(1) == self.size
    #logging.info('pred = {}'.format(pred.shape))
    #logging.info('ref = {}'.format(ref.shape))
    true_dist = pred.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, ref.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(ref.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(pred, Variable(true_dist, requires_grad=False))


