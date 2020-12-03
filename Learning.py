# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
from Model import save_checkpoint

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################
class Learning():
  def __init__(self, model, optim, optScheduler, criter, suffix, ol): 
    super(Learning, self).__init__()
    self.model = model
    self.optim = optim
    self.optScheduler = optScheduler
    self.criter = criter
    self.suffix = suffix
    self.max_steps = ol.max_steps
    self.max_epochs = ol.max_epochs
    self.validate_every = ol.validate_every
    self.save_every = ol.save_every
    self.report_every = ol.report_every
    self.keep_last_n = ol.keep_last_n
    self.step = optScheduler._step

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
      for i_batch, batch in enumerate(trainset):
        self.step += 1
        self.model.train()
        y_pred = self.model.forward(batch[0],batch[1],batch[2])
        loss = self.loss_fn(y_pred, batch[2])
        learning_total_loss += loss.item()
        loss_report += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        if self.report_every and self.step % self.report_every == 0: ### report
          loss_report = learning_total_loss
          step_report = self.step
          msec_report = time.time() 
          msec_per_batch = 1.0*(msec_report-msec_last_report)/(step_report-step_last_report)
          loss_per_batch = 1.0*(loss_report-loss_last_report)/(step_report-step_last_report)
          logging.info('Learning step:{} epoch:{} batch:{}/{} ms/batch: {:5.2f} lr:{:02.2f} loss/batch:{:5.2f}'.format(self.step, epoch, i_batch+1, len(trainset), msec_per_batch, self.optScheduler._rate, loss_per_batch))
          loss_last_report = loss_report
          step_last_report = step_report
          msec_last_report = msec_report

        if self.validate_every and self.step % self.validate_every == 0: ### validate
          if validset is not None:
            vloss = self.validate(validset)

        if self.save_every and self.step % self.save_every == 0: ### save
          save_checkpoint(self.model, self.optim, self.suffix, self.step, self.keep_last_n)

        if self.max_steps and self.step >= self.max_steps: ### stop by max_steps
          if validset is not None:
            vloss = self.validate(validset)
          save_checkpoint(self.model, self.optim, self.suffix, self.step, self.keep_last_n)
          return

      if self.max_epochs and epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset)
        save_checkpoint(self.model, self.optim, self.suffix, self.step, self.keep_last_n)
        return
    return

  def validate(self, validset):
    with torch.no_grad():
      model.eval()

    logging.info('Validation step {}'.format(self.step))
    return 0.0









