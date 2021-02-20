# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import torch
import math
from transformer.Model import prepare_source

def norm_length(l, alpha):
  if alpha == 0.0:
    return 1.0
  return (5+l)**alpha / (5+1)**alpha

##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, src_pre, tgt_pre, oi, device): 
    super(Inference, self).__init__()
    self.model = model
    self.src_pre = src_pre
    self.tgt_pre = tgt_pre
    self.beam_size = oi.beam_size
    self.max_size = oi.max_size
    self.n_best = oi.n_best
    self.alpha = oi.alpha
    self.format = oi.format
    self.idx_bos = tgt_pre.idx_bos
    self.idx_eos = tgt_pre.idx_eos
    self.Vt = len(tgt_pre)
    self.N = oi.n_best
    self.K = oi.beam_size
    self.device = device

    self.force_eos = torch.ones(len(self.tgt_pre), dtype=torch.float32, device=self.device) * float('Inf') #[Vt]
    self.force_eos[self.idx_eos] = 1.0
    self.next_wrds = torch.tensor([i for i in range(self.Vt)], dtype=int, device=self.device).view(1,-1) #[1,Vt]


  def translate(self, testset, output):
    logging.info('Running: inference')

    if output != '-':
      fh = open(output, 'w')
    else:
      fh = sys.stdout

    with torch.no_grad():
      self.model.eval()
      for pos, batch_src, _ in testset:
        ### encode batch
        src, self.msk_src = prepare_source(batch_src, self.tgt_pre.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
        self.z_src = self.model.encode(src, self.msk_src) #[bs,ls,ed]

        #for i in range(len(pos)):
        #  logging.debug('{}\n\t{}\n\t{}'.format(pos[i], src[i].tolist(), self.msk_src[i,0].tolist()))

        ### decode batch step-by-step
        if self.K == 1:
          finals = self.translate_greedy()
        else:
          finals = self.translate_beam()
        for b in range(len(finals)):
          for n, (hyp, logp) in enumerate(sorted(finals[b].items(), key=lambda kv: kv[1], reverse=True)):
            hyp = list(map(int,hyp.split(' ')))
            fh.write(self.format_hyp(pos[b],n,logp,hyp,batch_src[b]) + '\n')
            fh.flush()
            if n+1 >= self.N:
              break

    if output != '-':
      fh.close()


  def translate_greedy(self):
    bs =  self.z_src.shape[0]
    finals = [defaultdict() for i in range(bs)] #list with hyps reaching <eos> and overall score
    hyps = torch.ones([bs,1], dtype=int).to(self.device) * self.idx_bos #[bs,lt=1]
    logP = torch.zeros([bs,1], dtype=torch.float32).to(self.device)     #[bs,lt=1]
    next_wrds = self.next_wrds.repeat_interleave(repeats=bs, dim=0).view(bs*self.Vt,1) #[1,Vt] => [bs*1,Vt] => [bs*Vt,1]

    while True:
      lt = hyps.shape[1]

      ### DECODE ###
      msk_tgt = (1 - torch.triu(torch.ones((1, lt, lt), device=self.device), diagonal=1)).bool()
      y_next = self.model.decode(self.z_src, hyps, self.msk_src, msk_tgt=msk_tgt)[:,-1,:] #[bs,lt,Vt] => [bs,Vt]
 
      if lt == self.max_size - 1: #last extension
        y_next[:,] *= self.force_eos #all words are assigned -Inf except <eos> which keeps its logP

      next_logP = y_next.contiguous().view(bs*self.Vt,1)
      #logging.info('next_logP = {}'.format(next_logP.shape))

      ### EXPAND ###
      hyps_extended = hyps.repeat_interleave(repeats=self.Vt, dim=0) #[bs,lt] => [bs*Vt,lt]
      logP_extended = logP.repeat_interleave(repeats=self.Vt, dim=0) #[bs,lt] => [bs*Vt,lt]

      hyps_extended = torch.cat((hyps_extended, next_wrds), dim=-1).view(bs,self.Vt,lt+1) #[bs,Vt,lt+1]
      logP_extended = torch.cat((logP_extended, next_logP), dim=-1).view(bs,self.Vt,lt+1) #[bs,Vt,lt+1]
      lt = hyps_extended.shape[2] #new hypothesis length

      ### KEEP K-best expansions of each hypothesis I ###
      sum_logP = torch.sum(logP_extended,dim=2) #[bs,Vt]
      _, ind_best = torch.topk(sum_logP, k=1, dim=1) #[bs,1]

      hyps = torch.stack([hyps_extended[b][ind] for b,ind in enumerate(ind_best)], dim=0).contiguous().view(bs,lt) #[bs,1,lt] => [bs,lt]
      logP = torch.stack([logP_extended[b][ind] for b,ind in enumerate(ind_best)], dim=0).contiguous().view(bs,lt) #[bs,1,lt] => [bs,lt]

      ### FINALS ###
      index_of_finals = (hyps[:,-1]==self.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
      for b in index_of_finals:
        if len(finals[b]) < 1: ### not already finished
          hyp = ' '.join(map(str,hyps[b].tolist()))
          cst = sum(logP[b])
          if self.alpha:
            cst = cst / norm_length(hyps.shape[1],self.alpha)
          #logging.info('[FINAL b={}]\t{:6f}\t{}'.format(b,sum_logp,hyp))
          finals[b][hyp] = cst.item()
          logP[b,-1] = -float('Inf')

      if sum([len(d) for d in finals]) == bs:
        return finals


  def translate_beam(self):
    bs =  self.z_src.shape[0]
    finals = [defaultdict() for i in range(bs)] #list with hyps reaching <eos> and overall score
    hyps = torch.ones([bs,1], dtype=int).to(self.device) * self.idx_bos #[bs,lt=1]
    logP = torch.zeros([bs,1], dtype=torch.float32).to(self.device)     #[bs,lt=1]

    while True:
      I, lt = hyps.shape #I corresponds to: bs OR bs*K, lt is the hyp length
      #logging.info('hyps = {} logP = {}'.format(hyps.shape, logP.shape))

      if lt == 2:
        self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
        self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
        #logging.info('z_src = {} msk_src = {}'.format(self.z_src.shape, self.msk_src.shape))

      ### DECODE ###
      msk_tgt = (1 - torch.triu(torch.ones((1, lt, lt), device=self.device), diagonal=1)).bool()
      y_next = self.model.decode(self.z_src, hyps, self.msk_src, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      #logging.info('y_next = {}'.format(y_next.shape))
      if lt == self.max_size - 1: #last extension (force <eos>)
        y_next[:,] *= self.force_eos #all words are assigned -Inf except <eos> which keeps its logP

      next_logP = y_next.contiguous().view(-1,1) #[I*Vt,1]
      #logging.info('next_logP = {}'.format(next_logP.shape))
      next_wrds = self.next_wrds.repeat_interleave(repeats=I, dim=0).view(-1,1) #[1,Vt] => [1*I,Vt] => [I*Vt,1]
      #logging.info('next_wrds = {}'.format(next_wrds.shape))

      ### EXPAND ###
      hyps_extended = hyps.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
      logP_extended = logP.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
      #logging.info('hyps_extended = {} logP_extended = {}'.format(hyps_extended.shape, logP_extended.shape))

      hyps_extended = torch.cat((hyps_extended, next_wrds), dim=-1) #[I*Vt,lt+1]
      logP_extended = torch.cat((logP_extended, next_logP), dim=-1) #[I*Vt,lt+1]
      #logging.info('hyps_extended = {} logP_extended = {}'.format(hyps_extended.shape, logP_extended.shape))
      lt = hyps_extended.shape[1] #new hyp length

      ### KEEP K-best expansions of each hypothesis I ###
      hyps_extended = hyps_extended.contiguous().view(bs,-1,lt) #[bs,1*Vt,lt] or [bs,K*Vt,lt]
      logP_extended = logP_extended.contiguous().view(bs,-1,lt) #[bs,1*Vt,lt] or [bs,K*Vt,lt]
      #logging.info('hyps_extended = {} logP_extended = {}'.format(hyps_extended.shape, logP_extended.shape))
      sum_logP_extended = torch.sum(logP_extended,dim=2) #[bs,1*Vt] or [bs,K*Vt]

      _, kbest_inds = torch.topk(sum_logP_extended, k=self.K, dim=1) #both are [bs, K] (finds the K-best of dimension 1 (I*K)) no need to norm-length since all have same length
      hyps = torch.stack([hyps_extended[b][inds] for b,inds in enumerate(kbest_inds)], dim=0).contiguous().view(bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
      logP = torch.stack([logP_extended[b][inds] for b,inds in enumerate(kbest_inds)], dim=0).contiguous().view(bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
      #logging.info('hyps = {} logP = {}'.format(hyps.shape, logP.shape))
      #self.print_beam(bs, lt)

      ### FINALS ###
      index_of_finals = (hyps[:,-1]==self.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
      for i in index_of_finals:
        b = i//self.K
        if len(finals[b]) < self.K:
          hyp = ' '.join(map(str,hyps[i].tolist()))
          cst = sum(logP[i])
          if self.alpha:
            cst = cst / norm_length(hyps.shape[1],self.alpha)
          #logging.info('[FINAL b={}]\t{:6f}\t{}'.format(b,sum_logp,hyp))
          finals[b][hyp] = cst.item() # keep record of final hypothesis
          logP[i,-1] = -float('Inf') # force the hypothesis to disappear in next step

      if sum([len(d) for d in finals]) == bs*self.K:
        return finals

  def print_beam(self, bs, lt):
    hyps_bs_k = hyps.view(bs,self.K,lt)
    logP_bs_k = logP.view(bs,self.K,lt)
    for b in range(hyps_bs_k.shape[0]):
      for k in range(hyps_bs_k.shape[1]):
        logging.debug('batch {} beam {}\tlogP={:.6f}\t{}'.format(b, k, sum(logP_bs_k[b,k]), ' '.join([self.tgt_pre[t] for t in hyps_bs_k[b,k].tolist()]) ))


  def format_hyp(self, p, n, c, tgt_idx, src_idx): 
    #p is the position in the input sentence
    #n is the position in the nbest 
    #c is the hypothesis overall cost (sum_logP_norm)
    #tgt_idx hypothesis (list of ints)
    #src_idx source (list of ints)
    while src_idx[-1] == self.src_pre.idx_pad: # eliminate <pad> tokens from src_idx
      src_idx = src_idx[:-1]

    out = []
    for ch in self.format:
      if ch=='p':
        out.append("{}".format(p+1)) ### position in input file
      elif ch=='n':
        out.append("{}".format(n+1)) ### position in n-best order
      elif ch=='c':
        out.append("{:.6f}".format(c)) ### overall cost: sum(logP)
      ######################
      ### input sentence ###
      ######################
      elif ch=='s':
        out.append(' '.join([self.src_pre[idx] for idx in src_idx[1:-1]])) ### input sentence (tokenized)
      elif ch=='S':
        out.append(' '.join(self.src_pre.decode_list(src_idx[1:-1]))) ### input sentence (detokenized)
      elif ch=='j':
        out.append(' '.join(map(str,src_idx))) ### input sentence (idxs)
      #########################
      ### target hypothesis ###
      #########################
      elif ch=='t':
        out.append(' '.join([self.tgt_pre[idx] for idx in tgt_idx[1:-1]])) ### output sentence (tokenized)
      elif ch=='T':
        out.append(' '.join(self.tgt_pre.decode_list(tgt_idx[1:-1]))) ### output sentence (detokenized)
      elif ch=='i':
        out.append(' '.join(map(str,tgt_idx))) ### output sentence (idxs)

      else:
        logging.error('Invalid format option {} in {}'.format(ch,self.format))
        sys.exit()
    return '\t'.join(out)





