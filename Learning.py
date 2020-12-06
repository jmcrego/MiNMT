# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
from Model import save_checkpoint
from Optimizer import LabelSmoothing


def shift_batch_tgt(batch_tgt, batch_ltgt):
  batch_ref = []
  for i in range(len(batch_ltgt)):
    batch_ltgt[i] -= 1
    ref = batch_tgt[i].copy()
    batch_ref.append(ref)
    batch_ref[i].pop(0)
    batch_tgt[i].pop(batch_ltgt[i])
  return batch_tgt, batch_ref, batch_ltgt

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

  def learn(self, trainset, validset):
    logging.info('Running: learning')
    learning_total_loss = 0.
    loss_last_report = 0.
    step_last_report = 0
    msec_last_report = 0
    epoch = 0

    while True: #repeat epochs
      epoch += 1
      logging.info('Epoch {}'.format(epoch))

      trainset.shuffle()
      for i_batch, (batch_src, batch_tgt, batch_lsrc, batch_ltgt) in enumerate(trainset):
        batch_tgt, batch_ref, batch_ltgt = shift_batch_tgt(batch_tgt, batch_ltgt)
        self.model.train()
        y_pred = self.model.forward(batch_src,batch_tgt,batch_ref,batch_lsrc,batch_ltgt) #src, tgt, ref, lsrc, ltgt
        logging.info('y_pred = {}'.format(y_pred.shape))
        logging.info('batch_ref = {}'.format(torch.IntTensor(batch_ref).shape))
        loss = self.criter(y_pred, torch.IntTensor(batch_ref)) / sum(batch_ltgt) #or torch.sum(batch_ltgt)
        print(loss)
        sys.exit()
        learning_total_loss += loss.item()
        loss_report += loss.item()
        self.optScheduler.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optScheduler.step() #increments step, computes lr, updates model parameters

        if self.report_every and self.optScheduler._step % self.report_every == 0: ### report
          loss_report = learning_total_loss
          step_report = self.optScheduler._step
          msec_report = time.time()
          msec_per_batch = 1.0*(msec_report-msec_last_report)/(step_report-step_last_report)
          loss_per_batch = 1.0*(loss_report-loss_last_report)/(step_report-step_last_report)
          logging.info('Learning step:{} epoch:{} batch:{}/{} ms/batch: {:5.2f} lr:{:02.2f} loss/batch:{:5.2f}'.format(self.optScheduler._step, epoch, i_batch+1, len(trainset), msec_per_batch, self.optScheduler._rate, loss_per_batch))
          loss_last_report = loss_report
          step_last_report = step_report
          msec_last_report = msec_report

        if self.validate_every and self.optScheduler._step % self.validate_every == 0: ### validate
          if validset is not None:
            vloss = self.validate(validset)

        if self.save_every and self.optScheduler._step % self.save_every == 0: ### save
          save_checkpoint(self.suffix, self.model, self.optScheduler.optim, self.optScheduler._step, self.keep_last_n)

        if self.max_steps and self.optScheduler._step >= self.max_steps: ### stop by max_steps
          if validset is not None:
            vloss = self.validate(validset)
          save_checkpoint(self.suffix, self.model, self.OptScheduler.optim, self.optScheduler._step, self.keep_last_n)
          return

      if self.max_epochs and epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset)
        save_checkpoint(self.suffix, self.model, self.optScheduler.optim, self.optScheduler._step, self.keep_last_n)
        return
    return

  def validate(self, validset):
    with torch.no_grad():
      model.eval()

    logging.info('Validation step {}'.format(self.optScheduler._step))
    return 0.0









