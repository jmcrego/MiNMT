# -*- coding: utf-8 -*-

import torch

def build_AdamOptimizer(model, oo): 
  return torch.optim.Adam(model.parameters(), lr=oo.lr, betas=(oo.beta1, oo.beta2), eps=oo.eps)

class OptScheduler(): ### Adam optimizer with scheduler
  def __init__(self, optimizer, model_size, factor, warmup, step):
    super(OptScheduler, self).__init__()
    self.optimizer = optimizer #torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)
    self._step = step
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size
    self._rate = 0
        
  def step(self):
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()
        
  def rate(self, step = None):
    if step is None:
      step = self._step
    return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(torch.nn.Module):
  def __init__(self, size, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    self.criterion = torch.nn.KLDivLoss(reduction='sum')
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None

  def forward(self, x, target):
    assert x.size(1) == self.size
    true_dist = x.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(target.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(x, Variable(true_dist, requires_grad=False))




