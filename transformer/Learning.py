# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch
import time
from transformer.Model import save_checkpoint, prepare_source, prepare_target
try:
  from torch.utils.tensorboard import SummaryWriter
  tensorboard = True
except ImportError:
  tensorboard = False

##############################################################################################################
### Score ####################################################################################################
##############################################################################################################

class Score():
  def __init__(self):
    self.loss = 0.0
    self.ntok = 0
    self.nstep = 0
    self.start = time.time()

  def step(self, loss, ntok):
    self.loss += loss
    self.ntok += ntok
    self.nstep += 1

  def report(self):
    end = time.time()
    if self.ntok and self.nstep:
      loss_per_tok = self.loss / (1.0*self.nstep)
      steps_per_sec = self.nstep / (end - self.start)
      return loss_per_tok, steps_per_sec
    logging.warning('Requested report after 0 steps optimised')
    return 0.0, 0

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################

class Learning():
  def __init__(self, model, optScheduler, criter, suffix, idx_pad, inference_valid, ol):
    super(Learning, self).__init__()
    self.dnet = ol.dnet
    self.model = model
    self.optScheduler = optScheduler
    self.criter = criter #LabelSmoothing+KLDivLoss or NLLLoss
    self.suffix = suffix
    self.max_steps = ol.max_steps
    self.max_epochs = ol.max_epochs
    self.validate_every = ol.validate_every
    self.save_every = ol.save_every
    self.report_every = ol.report_every
    self.keep_last_n = ol.keep_last_n
    self.clip = ol.clip
    self.accum_n_batchs = ol.accum_n_batchs
    self.idx_pad = idx_pad
    self.inference = inference_valid

    if tensorboard:
      self.writer = SummaryWriter(log_dir=ol.dnet, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='')

  def learn(self, trainset, validset, device):
    logging.info('Running: learning')
    n_epoch = 0
    self.optScheduler.optimizer.zero_grad() ### sets gradients to zero
    while True: #repeat epochs
      n_epoch += 1
      logging.info('Epoch {}'.format(n_epoch))
      n_batch = 0
      ntok_in_step = 0
      loss_accum = 0.0
      score = Score()
      for batch_pos, batch_idxs in trainset:
        batch_src = batch_idxs[0]
        batch_tgt = batch_idxs[1]
        n_batch += 1
        self.model.train()
        ###
        ### forward
        ###
        src, msk_src = prepare_source(batch_src, self.idx_pad, device)
        tgt, ref, msk_tgt = prepare_target(batch_tgt, self.idx_pad, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt) #no log_softmax is applied
        ntok_in_batch = torch.sum(ref != self.idx_pad)
        ntok_in_step += ntok_in_batch
        ###
        ### compute loss
        ###
        loss = self.criter(pred, ref) / ntok_in_batch / self.accum_n_batchs #sum of losses in batch (normalized by tokens in batch) (n batchs will be accumulated before model update, so i normalize by n batchs)
        loss_accum += loss.item()
        ###
        ### compute/accumulate gradients (accumulate gradients until step() is called)
        ###
        loss.backward()
        if n_batch % self.accum_n_batchs == 0: #waits for n backward steps
          ###
          ### optimize (update model)
          ###
          if self.clip > 0.0: ### clip gradients norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
          self.optScheduler.step() ### updates model parameters after incrementing step and updating lr
          self.optScheduler.optimizer.zero_grad() ### sets gradients to zero for next update
          ### add score
          score.step(loss_accum, ntok_in_step)
          ntok_in_step = 0
          loss_accum = 0.0
          ###
          ### report
          ###
          if self.report_every and self.optScheduler._step and self.optScheduler._step % self.report_every == 0:  ### first _step is 0
            loss_per_tok, steps_per_sec = score.report()
            logging.info('Learning step: {} epoch: {} batch: {} steps/sec: {:.2f} lr: {:.6f} Loss: {:.3f}'.format(self.optScheduler._step, n_epoch, n_batch, steps_per_sec, self.optScheduler._rate, loss_per_tok))
            score = Score()
            if tensorboard:
              self.writer.add_scalar('Loss/train', loss_accum, self.optScheduler._step)
              self.writer.add_scalar('LearningRate', self.optScheduler._rate, self.optScheduler._step)
          ###
          ### validate
          ###
          if self.validate_every and self.optScheduler._step and self.optScheduler._step % self.validate_every == 0: 
            if validset is not None:
              vloss = self.validate(validset, device)
          ###
          ### save
          ###
          if self.save_every and self.optScheduler._step and self.optScheduler._step % self.save_every == 0: 
            save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
          ###
          ### stop by max_steps
          ###
          if self.max_steps and self.optScheduler._step and self.optScheduler._step >= self.max_steps: 
            if validset is not None:
              vloss = self.validate(validset, device)
            save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
            logging.info('Learning STOP by [steps={}]'.format(self.optScheduler._step))
            return
      ###
      ### stop by max_epochs
      ###
      if self.max_epochs and n_epoch >= self.max_epochs: ### stop by max_epochs
        if validset is not None:
          vloss = self.validate(validset, device)
        save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
        logging.info('Learning STOP by [epochs={}]'.format(n_epoch))
        return

  def validate(self, validset, device):
    tic = time.time()
    valid_loss = 0.0
    n_batch = 0
    with torch.no_grad():
      self.model.eval()
      for batch_pos, batch_idxs in validset:
        batch_src = batch_idxs[0]
        batch_tgt = batch_idxs[1]
        n_batch += 1
        src, msk_src = prepare_source(batch_src, self.idx_pad, device)
        tgt, ref, msk_tgt = prepare_target(batch_tgt, self.idx_pad, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt) #no log_softmax is applied
        loss = self.criter(pred, ref) ### batch loss
        valid_loss += loss.item() / torch.sum(ref != self.idx_pad)
        if n_batch == 1:
          print_pos_src_tgt_hyp_ref(pred[0], batch_pos[0], src[0], tgt[0], ref[0])

    toc = time.time()
    loss = 1.0*valid_loss/n_batch if n_batch else 0.0
    bleu = self.translate_valid(validset)
    logging.info('Validation step: {} #batchs: {} sec: {:.2f} bleu: {:.2f} loss: {:.3f}'.format(self.optScheduler._step, n_batch, toc-tic, bleu, loss))
    if tensorboard:
      self.writer.add_scalar('Loss/valid', loss, self.optScheduler._step)
    return loss

  def translate_valid(self, validset):
    fhyp = '{}/valid_{:08d}.out'.format(self.dnet,self.optScheduler._step)
    self.inference.translate(validset, fhyp)    
    return 0.0

def print_pos_src_tgt_hyp_ref(pred, pos, src, tgt, ref):
  hyp = torch.nn.functional.log_softmax(pred, dim=-1) #[lt,Vt]
  _, ind = torch.topk(hyp, k=1, dim=-1) #[lt,1]
  logging.info('POS: {}'.format(pos))
  logging.info('SRC: ' + ' '.join(['{: ^5}'.format(t) for t in src.tolist()]))
  #logging.info('TGT: ' + ' '.join(['{: ^5}'.format(t) for t in tgt.tolist()]))
  logging.info('HYP: ' + ' '.join(['{: ^5}'.format(t) for t in ind.squeeze(-1).tolist()]))
  logging.info('REF: ' + ' '.join(['{: ^5}'.format(t) for t in ref.tolist()]))

