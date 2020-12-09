# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from Model import save_checkpoint
from Optimizer import LabelSmoothing

def prepare_input(batch_src, batch_tgt, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] #as is
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  if batch_tgt is None:
    return src, None, None
  tgt = [torch.tensor(seq[:-1]) for seq in batch_tgt] #delete <eos>
  tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=idx_pad).to(device) 
  ref = [torch.tensor(seq[1:])  for seq in batch_tgt] #delete <bos>
  ref = torch.nn.utils.rnn.pad_sequence(ref, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  msk_tgt = (tgt != idx_pad).unsqueeze(-2) & (1 - torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)).bool() #[bs,lt,lt]
  return src, tgt, ref, msk_src, msk_tgt

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

  def learn(self, trainset, validset, idx_pad, device, max_length):
    logging.info('Running: learning')
    loss_report = 0.
    step_report = 0
    msec_report = time.time()
    epoch = 0

    while True: #repeat epochs
      epoch += 1
      logging.info('Epoch {}'.format(epoch))

      trainset.shuffle()
      n_batch = 0
      for batch_src, batch_tgt in trainset:
#        if len(batch_src[-1]) != 10 or len(batch_tgt[-1]) != 10:
#          continue
        if max_length > 0 and (len(batch_src[-1]) > max_length or len(batch_tgt[-1]) > max_length): 
          logging.debug('skipped batch with src/tgt size {}/{}'.format(len(batch_src[-1]), len(batch_tgt[-1])))
          continue
        n_batch += 1
        self.model.train()

        src, tgt, ref, msk_src, msk_tgt = prepare_input(batch_src, batch_tgt, idx_pad, device)
#        print('src = {}'.format(src.shape))
#        for l in src:
#          print(' '.join(map(str,l.tolist())))
#        print('tgt = {}'.format(tgt.shape))
#        for l in tgt:
#          print(' '.join(map(str,l.tolist())))

        pred = self.model.forward(src, tgt, msk_src, msk_tgt)
#        print('pred = {}'.format(pred.shape))
#        print('ref = {}'.format(ref.shape))
#        for l in ref:
#          print(' '.join(map(str,l.tolist())))

        loss_batch = self.criter(pred, ref)
        loss_token = loss_batch / torch.sum(ref != idx_pad)
        #print(loss_batch.item(), loss_token.item())

        loss_report += loss_token.item()
        step_report += 1
        self.optScheduler.optimizer.zero_grad()                                      #sets gradients to zero
        loss_batch.backward()                                                        #computes gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm) #clip gradients
        self.optScheduler.step()                                                     #updates model parameters after incrementing step and updating lr

        if self.report_every and self.optScheduler._step % self.report_every == 0: ### report
          msec_per_batch = 1000.0*(time.time()-msec_report)/step_report
          loss = 1.0*loss_report/step_report
          logging.info('Learning step:{} epoch:{} batch:{}/{} ms/batch:{:.2f} lr:{:.8f} loss:{:.3f}'.format(self.optScheduler._step, epoch, n_batch, len(trainset), msec_per_batch, self.optScheduler._rate, loss))
          loss_report = 0
          step_report = 0
          msec_report = time.time()

        if self.validate_every and self.optScheduler._step % self.validate_every == 0: ### validate
          if validset is not None:
            vloss = self.validate(validset, idx_pad, device, max_length)

        if self.save_every and self.optScheduler._step % self.save_every == 0: ### save
          save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)

        if self.max_steps and self.optScheduler._step >= self.max_steps: ### stop by max_steps
          if validset is not None:
            vloss = self.validate(validset, idx_pad, device, max_length)
          save_checkpoint(self.suffix, self.model, self.OptScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
          return

      logging.info('End of epoch {} after {} batches out of {} (remaining were skipped by length)'.format(epoch,n_batch,len(trainset)))

      if self.max_epochs and epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset, idx_pad, device, max_length)
        save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
        return
    return

  def validate(self, validset, idx_pad, device, max_length):
    with torch.no_grad():
      self.model.eval()

      tic = time.time()
      valid_loss = 0.
      n_batch = 0
      for batch_src, batch_tgt in validset:
        if max_length > 0 and (len(batch_src[-1]) > max_length or len(batch_tgt[-1]) > max_length): 
          logging.debug('skipped batch with src/tgt size {}/{}'.format(len(batch_src[-1]), len(batch_tgt[-1])))
          continue
        n_batch += 1
        src, tgt, ref = prepare_input(batch_src, batch_tgt, idx_pad, device)
        pred = self.model.forward(src, tgt)
        loss_batch = self.criter(pred, ref)
        loss_token = loss_batch / torch.sum(ref != idx_pad)
        valid_loss += loss_token.item()

    toc = time.time()
    loss = valid_loss/n_batch if n_batch else 0.0
    logging.info('Validation #batchs:{} sec:{:.2f} loss:{:.3f}'.format(n_batch, toc-tic, loss))
    return loss_per_batch









