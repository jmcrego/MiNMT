# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from Model import save_checkpoint
from Optimizer import LabelSmoothing

def shift_batch_tgt_ref(src, tgt, idx_pad):
  #batch_tgt os [bs, sl]
  batch_ref = trg[:, 1:] #does not consider the first in sl
  batch_trg = trg[:, :-1] #does not consider the last in sl

  batch_ref = []
  for i in range(len(batch_tgt)):
    ref = batch_tgt[i].copy()
    batch_ref.append(ref)
    batch_ref[i].pop(0)
    batch_tgt[i].pop(batch_ltgt[i])
  return batch_tgt, batch_ref

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################
class Learning():
  def __init__(self, model, optScheduler, criter, suffix, ol): 
    super(Learning, self).__init__()
    self.model = model
    self.optScheduler = optScheduler
    self.criter = criter #label_smoothing
    self.suffix = suffix
    self.max_steps = ol.max_steps
    self.max_epochs = ol.max_epochs
    self.validate_every = ol.validate_every
    self.save_every = ol.save_every
    self.report_every = ol.report_every
    self.keep_last_n = ol.keep_last_n
    self.clip_grad_norm = ol.clip_grad_norm

  def learn(self, trainset, validset, idx_pad, device):
    logging.info('Running: learning')
    loss_report = 0.
    step_report = 0
    msec_report = time.time()
    epoch = 0

    while True: #repeat epochs
      epoch += 1
      logging.info('Epoch {}'.format(epoch))

      trainset.shuffle()
      for i_batch, (batch_src, batch_tgt) in enumerate(trainset):
        src = [torch.tensor(seq)      for seq in batch_src] #as is
        tgt = [torch.tensor(seq[:-1]) for seq in batch_tgt] #delete <eos>
        ref = [torch.tensor(seq[1:])  for seq in batch_tgt] #delete <bos>
        src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
        tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=idx_pad).to(device)
        ref = torch.nn.utils.rnn.pad_sequence(ref, batch_first=True, padding_value=idx_pad).to(device)
        self.model.train()
        pred = self.model.forward(src, tgt) 
        loss = self.criter(pred, ref) / torch.sum(ref != idx_pad)
        loss_report += loss.item()
        step_report += 1
        self.optScheduler.optimizer.zero_grad()                                      #sets gradients to zero
        loss.backward()                                                              #computes gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm) #clip gradients
        self.optScheduler.step()                                                     #updates model parameters after incrementing step and updating lr

        if self.report_every and self.optScheduler._step % self.report_every == 0: ### report
          msec_per_batch = 1000.0*(time.time()-msec_report)/step_report
          loss_per_batch = 1.0*loss_report/step_report
          logging.info('Learning step:{} epoch:{} batch:{}/{} ms/batch:{:.2f} lr:{:.8f} loss/batch:{:.3f}'.format(self.optScheduler._step, epoch, i_batch+1, len(trainset), msec_per_batch, self.optScheduler._rate, loss_per_batch))
          loss_report = 0
          step_report = 0
          msec_report = time.time()

        if self.validate_every and self.optScheduler._step % self.validate_every == 0: ### validate
          if validset is not None:
            vloss = self.validate(validset)

        if self.save_every and self.optScheduler._step % self.save_every == 0: ### save
          save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)

        if self.max_steps and self.optScheduler._step >= self.max_steps: ### stop by max_steps
          if validset is not None:
            vloss = self.validate(validset)
          save_checkpoint(self.suffix, self.model, self.OptScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
          return

      if self.max_epochs and epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset)
        save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
        return
    return

  def validate(self, validset):
    with torch.no_grad():
      model.eval()

    logging.info('Validation step {}'.format(self.optScheduler._step))
    return 0.0









