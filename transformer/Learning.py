# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from transformer.Model import save_checkpoint
#from Optimizer import LabelSmoothing

def prepare_input(batch_src, batch_tgt, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] #as is
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  #if batch_tgt is None:
  #  return src, None, None, msk_src, None
  tgt = [torch.tensor(seq[:-1]) for seq in batch_tgt] #delete <eos>
  tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=idx_pad).to(device) 
  ref = [torch.tensor(seq[1:])  for seq in batch_tgt] #delete <bos>
  ref = torch.nn.utils.rnn.pad_sequence(ref, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  msk_tgt = (tgt != idx_pad).unsqueeze(-2) & (1 - torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)).bool() #[bs,lt,lt]
  return src, tgt, ref, msk_src, msk_tgt

##############################################################################################################
### Score ####################################################################################################
##############################################################################################################

class Score():
  def __init__(self):
    #global
    self.nsteps = 0
    self.loss = 0.
    self.ntoks = 0
    self.msec_epoch = time.time()
    #report
    self.loss_report = 0.
    self.ntoks_report = 0
    self.nsteps_report = 0
    self.msec_report = self.msec_epoch


  def step(self, sum_loss_batch, ntoks_batch):
    #global
    self.nsteps += 1
    self.loss += sum_loss_batch
    self.ntoks += ntoks_batch
    #report
    self.loss_report += sum_loss_batch
    self.ntoks_report += ntoks_batch
    self.nsteps_report += 1

  def report(self):
    tnow = time.time()
    if self.ntoks_report and self.nsteps_report:
      loss_per_tok = self.loss_report / (1.0*self.ntoks_report)
      ms_per_step = 1000.0 * (tnow - self.msec_report) / (1.0*self.nsteps_report)
    else:
      loss_per_tok = 0.
      ms_per_step = 0.
      logging.warning('Requested report after 0 tokens optimised')
    #initialize for next report
    self.loss_report = 0.
    self.ntoks_report = 0
    self.nsteps_report = 0
    self.msec_report = tnow
    return loss_per_tok, ms_per_step

  def epoch(self):
    tnow = time.time()
    if self.ntoks and self.nsteps:
      loss_per_tok = self.loss / (1.0*self.ntoks)
      ms_epoch = 1000.0 * tnow - self.msec_epoch
    else:
      loss_per_tok = 0.
      ms_epoch = 0.
      logging.warning('Requested epoch report after 0 tokens optimised')
    #no need to initialize
    return loss_per_tok, ms_epoch

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################

class Learning():
  def __init__(self, model, optScheduler, criter, suffix, idx_pad, ol): 
    super(Learning, self).__init__()
    self.model = model
    self.optScheduler = optScheduler
    self.criter = criter #LabelSmoothing
    self.suffix = suffix
    self.max_steps = ol.max_steps
    self.max_epochs = ol.max_epochs
    self.validate_every = ol.validate_every
    self.save_every = ol.save_every
    self.report_every = ol.report_every
    self.keep_last_n = ol.keep_last_n
    self.clip_grad_norm = ol.clip_grad_norm
    self.idx_pad = idx_pad

  def learn(self, trainset, validset, device):
    logging.info('Running: learning')
    n_epoch = 0
    while True: #repeat epochs
      n_epoch += 1
      logging.info('Epoch {}'.format(n_epoch))
      trainset.shuffle()
      n_batch = 0
      score = Score()
      for _, batch_src, batch_tgt in trainset:
        n_batch += 1
        self.model.train()
        ### predict (forward)
        src, tgt, ref, msk_src, msk_tgt = prepare_input(batch_src, batch_tgt, self.idx_pad, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt)
        ### compute loss
        loss_batch = self.criter(pred, ref)
        loss_token = loss_batch / torch.sum(ref != self.idx_pad)
        ### optimize
        self.optScheduler.optimizer.zero_grad()                                        ### sets gradients to zero
        loss_token.backward()                                                          ### computes gradients
        if self.clip_grad_norm > 0.0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm) ### clip gradients to clip_grad_norm
        self.optScheduler.step()                                                       ### updates model parameters after incrementing step and updating lr
        ### accumulate score
        score.step(loss_batch.item(), torch.sum(ref != self.idx_pad))

        if self.report_every and self.optScheduler._step % self.report_every == 0: ### report
          loss_per_tok, ms_per_step = score.report()
          logging.info('Learning step:{} epoch:{} batch:{}/{} ms/batch:{:.2f} lr:{:.6f} loss/tok:{:.3f}'.format(self.optScheduler._step, n_epoch, n_batch, len(trainset), ms_per_step, self.optScheduler._rate, loss_per_tok))

        if self.validate_every and self.optScheduler._step % self.validate_every == 0: ### validate
          if validset is not None:
            vloss = self.validate(validset, device)

        if self.save_every and self.optScheduler._step % self.save_every == 0: ### save
          save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)

        if self.max_steps and self.optScheduler._step >= self.max_steps: ### stop by max_steps
          if validset is not None:
            vloss = self.validate(validset, device)
          save_checkpoint(self.suffix, self.model, self.OptScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
          logging.info('Learning: STOP by [steps={}]'.format(self.optScheduler._step))
          return

      loss_per_tok, ms_epoch = score.epoch()
      logging.info('End of epoch {} #batches:{} loss/tok:{:.3f} sec:{:.2f}'.format(n_epoch,n_batch,loss_per_tok,ms_epoch/1000.0))

      if self.max_epochs and n_epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset, device)
        save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
        logging.info('Learning: STOP by [epochs={}]'.format(n_epoch))
        return
    return

  def validate(self, validset, device):
    tic = time.time()
    with torch.no_grad():
      self.model.eval()
      valid_loss = 0.
      n_batch = 0
      for _, batch_src, batch_tgt in validset:
        n_batch += 1
        src, tgt, ref, msk_src, msk_tgt = prepare_input(batch_src, batch_tgt, self.idx_pad, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt)
        loss = self.criter(pred, ref) ### batch loss
        valid_loss += loss.item() / torch.sum(ref != self.idx_pad)

    toc = time.time()
    loss = 1.0*valid_loss/n_batch if n_batch else 0.0
    logging.info('Validation learning steps:{} #batchs:{} sec:{:.2f} loss:{:.3f}'.format(self.optScheduler._step, n_batch, toc-tic, loss))
    return loss









