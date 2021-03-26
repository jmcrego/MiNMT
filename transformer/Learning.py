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

def pad_prefix(ref, idx_sep, idx_pad):
  #ref is [bs, lt]
  inds_sep = (ref == idx_sep).nonzero(as_tuple=True)[1] #[bs] position of idx_sep tokens in ref (one per line)
  #logging.info('inds_sep = {}'.format(inds_sep))
  assert ref.shape[0] == inds_sep.shape[0], 'references must contain one and no more than one idx_sep tokens {}!={}'.format(ref.shape,inds_sep.shape)
  seqs_sep = [torch.ones([l+1], dtype=torch.long) for l in inds_sep]
  #logging.info('seqs_sep = {}'.format(seqs_sep))
  padding = torch.nn.utils.rnn.pad_sequence(seqs_sep, batch_first=True, padding_value=0).to(ref.device)
  #logging.info('padding = {}'.format(padding))
  if padding.shape[1] < ref.shape[1]:
    extend = torch.zeros([ref.shape[0], ref.shape[1]-padding.shape[1]], dtype=torch.long, device=ref.device)
    padding = torch.cat((padding, extend), 1)
  #logging.info('padding = {}'.format(padding))
  ref = torch.where(padding==1,idx_pad,ref)
  #logging.info('ref = {}'.format(ref))
  return ref

##############################################################################################################
### Score ####################################################################################################
##############################################################################################################

class Score():
  def __init__(self):
    self.sum_loss_report = 0.
    self.sum_toks_report = 0
    self.nsteps_report = 0
    self.n_msk = 0
    self.n_ok_msk = 0
    self.start_report = time.time()

  def step(self, sum_loss_batch, ntok_batch, pred, gold, idx_msk):
    self.sum_loss_report += sum_loss_batch
    self.sum_toks_report += ntok_batch
    self.nsteps_report += 1
    n_msk, n_ok_msk = self.eval_msk(pred, gold, idx_msk)
    self.n_msk += n_msk 
    self.n_ok_msk += n_ok_msk 

  def report(self):
    end_report= time.time()
    if self.sum_toks_report and self.nsteps_report:
      loss_per_tok = self.sum_loss_report / (1.0*self.sum_toks_report)
      steps_per_sec = self.nsteps_report / (end_report - self.start_report)
      if n_msk > 0:
        logging.info('n_msk: {} acc_msk: {:.2f}'.format(self.n_msk, 100.0*self.n_ok_msk/self.n_msk))
      return loss_per_tok, steps_per_sec
    logging.warning('Requested report after 0 tokens optimised')
    return 0., 0

  def eval_msk(self, pred, gold, idx_msk):
    bs, lt, ed = pred.shape
    gold = gold.contiguous().view([bs*lt])
    pred = pred.contiguous().view([bs*lt,-1])
    inds_gold_msk = (gold==idx_msk).nonzero(as_tuple=True)[0] #[n] indexs i of gold where gold[i]=idx_msk
    n_ok_msk = 0
    n_msk = torch.numel(inds_gold_msk)
    if n_msk > 0:
      _, inds_pred = torch.topk(pred, k=1, dim=1)
      inds_pred_msk = inds_pred[inds_gold_msk].squeeze()
      n_ok_msk = torch.sum(inds_pred_msk==idx_msk) #.nonzero(as_tuple=False)
    return n_msk, n_ok_msk

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################

class Learning():
  def __init__(self, model, optScheduler, criter, suffix, idx_pad, idx_sep, idx_msk, ol):
    super(Learning, self).__init__()
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
    self.mask_prefix = ol.mask_prefix
    self.pad_prefix = ol.pad_prefix
    self.idx_pad = idx_pad    
    self.idx_sep = idx_sep
    self.idx_msk = idx_msk

    if tensorboard:
      self.writer = SummaryWriter(log_dir=ol.dnet, comment='', purge_step=None, max_queue=10, flush_secs=60, filename_suffix='')

  def learn(self, trainset, validset, device):
    logging.info('Running: learning')
    n_epoch = 0
    while True: #repeat epochs
      n_epoch += 1
      logging.info('Epoch {}'.format(n_epoch))
      n_batch = 0
      score = Score()
      for batch_pos, [batch_src, batch_tgt] in trainset:
        n_batch += 1
        self.model.train()
        ###
        ### forward
        ###
        src, msk_src = prepare_source(batch_src, self.idx_pad, device)
        tgt, ref, msk_tgt = prepare_target(batch_tgt, self.idx_pad, self.idx_sep, self.idx_msk, self.mask_prefix, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt) #no log_softmax is applied
        ###
        ### compute loss
        ###
        if self.pad_prefix: #do not compute loss over prefix tokens
          ref = pad_prefix(ref, self.idx_sep, self.idx_pad)
        loss_batch = self.criter(pred, ref) #sum of losses in batch
        loss_token = loss_batch / torch.sum(ref != self.idx_pad) #ntok_batch
        ### optimize
        ###
        self.optScheduler.optimizer.zero_grad() ### sets gradients to zero
        loss_token.backward() ### computes gradients
        if self.clip > 0.0: ### clip gradients norm
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optScheduler.step() ### updates model parameters after incrementing step and updating lr
        ###
        ### accumulate score
        ###
        score.step(loss_batch.item(), torch.sum(ref!=self.idx_pad), pred, ref, self.idx_msk)
        ###
        ### report
        ###
        if self.report_every and self.optScheduler._step % self.report_every == 0: 
          loss_per_tok, steps_per_sec = score.report()
          logging.info('Learning step: {} epoch: {} batch: {} steps/sec: {:.2f} lr: {:.6f} Loss: {:.3f}'.format(self.optScheduler._step, n_epoch, n_batch, steps_per_sec, self.optScheduler._rate, loss_per_tok))
          score = Score()
          if tensorboard:
            self.writer.add_scalar('Loss/train', loss_token.item(), self.optScheduler._step)
            self.writer.add_scalar('LearningRate', self.optScheduler._rate, self.optScheduler._step)
        ###
        ### validate
        ###
        if self.validate_every and self.optScheduler._step % self.validate_every == 0: 
          if validset is not None:
            vloss = self.validate(validset, device)
        ###
        ### save
        ###
        if self.save_every and self.optScheduler._step % self.save_every == 0: 
          save_checkpoint(self.suffix, self.model, self.optScheduler.optimizer, self.optScheduler._step, self.keep_last_n)
        ###
        ### stop by max_steps
        ###
        if self.max_steps and self.optScheduler._step >= self.max_steps: 
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
    valid_loss = 0.
    n_batch = 0
    with torch.no_grad():
      self.model.eval()
      for batch_pos, batch_idxs in validset:
        batch_src = batch_idxs[0]
        batch_tgt = batch_idxs[1]
        n_batch += 1
        src, msk_src = prepare_source(batch_src, self.idx_pad, device)
        tgt, ref, msk_tgt = prepare_target(batch_tgt, self.idx_pad, self.idx_sep, self.idx_msk, self.mask_prefix, device)
        pred = self.model.forward(src, tgt, msk_src, msk_tgt) #no log_softmax is applied
        if self.pad_prefix:
          ref = pad_prefix(ref, self.idx_sep, self.idx_pad)
        loss = self.criter(pred, ref) ### batch loss
        valid_loss += loss.item() / torch.sum(ref != self.idx_pad)
        if n_batch == 1:
          print_pos_src_tgt_hyp_ref(pred[0], batch_pos[0], src[0], tgt[0], ref[0])

    toc = time.time()
    loss = 1.0*valid_loss/n_batch if n_batch else 0.0
    logging.info('Validation step: {} #batchs: {} sec: {:.2f} loss: {:.3f}'.format(self.optScheduler._step, n_batch, toc-tic, loss))
    if tensorboard:
      self.writer.add_scalar('Loss/valid', loss, self.optScheduler._step)
    return loss

def print_pos_src_tgt_hyp_ref(pred, pos, src, tgt, ref):
  hyp = torch.nn.functional.log_softmax(pred, dim=-1) #[lt,Vt]
  _, ind = torch.topk(hyp, k=1, dim=-1) #[lt,1]
  logging.info('POS: {}'.format(pos))
  logging.info('SRC: ' + ' '.join(['{: ^5}'.format(t) for t in src.tolist()]))
  logging.info('TGT: ' + ' '.join(['{: ^5}'.format(t) for t in tgt.tolist()]))
  logging.info('HYP: ' + ' '.join(['{: ^5}'.format(t) for t in ind.squeeze(-1).tolist()]))
  logging.info('REF: ' + ' '.join(['{: ^5}'.format(t) for t in ref.tolist()]))

