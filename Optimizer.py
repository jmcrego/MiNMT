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

  def forward(self, pred, gold):
    #pred is [bs, lt, Vt]
    #gold is [bs, lt]
    assert pred.size(0) == gold.size(0)
    assert pred.size(1) == gold.size(1)
    assert pred.size(2) == self.size
    pred = pred.contiguous().view(-1, pred.size(-1)) #[bs*lt, Vt]
    gold = gold.contiguous().view(-1) #gold is [bs*lt]

    #pred is [bs*lt, Vt]
    #gold is [bs*lt]
    #eps = 0.1
    #n_class = pred.size(1)
    #one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)

    #one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    #log_prb = F.log_softmax(pred, dim=1)
    #non_pad_mask = gold.ne(trg_pad_idx)
    #loss = -(one_hot * log_prb).sum(dim=1)
    #loss = loss.masked_select(non_pad_mask).sum()  # average later

    true_dist = pred.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, gold.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(gold.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(pred, Variable(true_dist, requires_grad=False))


