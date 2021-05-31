# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import torch
import math
from transformer.Model import prepare_source, prepare_source_cross

def norm_length(l, alpha):
  if alpha == 0.0:
    return 1.0
  return (5+l)**alpha / (5+1)**alpha

##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, src_voc, tgt_voc, oi, model_type, device): 
    super(Inference, self).__init__()
    self.model = model
    self.model_type = model_type
    self.src_voc = src_voc
    self.tgt_voc = tgt_voc
    self.Vt = len(tgt_voc)
    self.max_size = oi.max_size
    self.alpha = oi.alpha
    self.format = oi.format
    self.K = oi.beam_size
    self.N = oi.n_best
    self.prefix = oi.prefix is not None
    self.device = device
    self.next_wrds = torch.tensor([i for i in range(self.Vt)], dtype=int, device=self.device).view(1,-1) #[1,Vt]


  def translate(self, testset, output):
    logging.info('Running: inference')
    dhyps = {}

    if output != '-':
      fh = open(output, 'w')
    else:
      fh = sys.stdout

    with torch.no_grad():
      self.model.eval()
      for pos, batch_idxs in testset:

        ##############
        ### ENCODE ###
        ##############
        if self.model_type == 's_sc':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          self.z_src = self.model.encode(src, self.msk_src) #[bs,ls,ed]

        elif self.model_type == 's_s_scc_scc':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          xsrc, msk_xsrc = prepare_source(batch_idxs[1], self.src_voc.idx_pad, self.device) #xsrc is [bs, lxs] msk_xsrc is [bs,1,lxs]
          xtgt, self.msk_xtgt = prepare_source(batch_idxs[2], self.tgt_voc.idx_pad, self.device) #xtgt is [bs, lxt] msk_xtgt is [bs,1,lxt]
          self.z_src, self.z_xtgt = self.model.encode(src, xsrc, xtgt, self.msk_src, msk_xsrc, self.msk_xtgt) #[bs,ls,ed] [bs,lxt,ed]

        elif self.model_type == '2nmt_2c':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          xsrc, msk_xsrc = prepare_source(batch_idxs[1], self.src_voc.idx_pad, self.device) #xsrc is [bs, lxs] msk_xsrc is [bs,1,lxs]
          xtgt, self.msk_xtgt = prepare_source(batch_idxs[2], self.tgt_voc.idx_pad, self.device) #xtgt is [bs, lxt] msk_xtgt is [bs,1,lxt]
        self.z_src, self.z_xtgt = self.model.encode(src, xsrc, xtgt, self.msk_src, msk_xsrc, self.msk_xtgt) #[bs,ls,ed] [bs,lxt,ed]

        elif self.model_type == 'sxs_sc':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          xtgt, self.msk_xtgt = prepare_source(batch_idxs[1], self.tgt_voc.idx_pad, self.device) #xtgt is [bs, lxt] msk_xtgt is [bs,1,lxt]
          self.z_src, self.z_xtgt = self.model.encode(src, xtgt, self.msk_src, self.msk_xtgt) #[bs,ls,ed] [bs,lxt,ed]

        elif self.model_type == 'sxsc_sc':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          xtgt, self.msk_xtgt = prepare_source(batch_idxs[1], self.tgt_voc.idx_pad, self.device) #xtgt is [bs, lxt] msk_xtgt is [bs,1,lxt]
          self.z_src, self.z_xtgt = self.model.encode(src, xtgt, self.msk_src, self.msk_xtgt) #[bs,ls,ed] [bs,lxt,ed]

        elif self.model_type == 's_s_scc':
          src, self.msk_src = prepare_source(batch_idxs[0], self.src_voc.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
          xtgt, self.msk_xtgt = prepare_source(batch_idxs[1], self.tgt_voc.idx_pad, self.device) #xtgt is [bs, lxt] msk_xtgt is [bs,1,lxt]
          self.z_src, self.z_xtgt = self.model.encode(src, xtgt, self.msk_src, self.msk_xtgt) #[bs,ls,ed] [bs,lxt,ed]


        if self.prefix: ### if prefix the last is the prefix 
          self.batch_pre, _ = prepare_source(batch_idxs[-1], self.tgt_voc.idx_pad, self.device)  #pre is [bs, lp]
        else:
          self.batch_pre = None

        ### decode (step-by-step) ###
        finals = self.traverse_beam()
        ### eoutput
        for b in range(len(finals)):
          for n, (hyp, logp) in enumerate(sorted(finals[b].items(), key=lambda kv: kv[1], reverse=True)):
            hyp = list(map(int,hyp.split(' ')))
            out, hyp = self.format_hyp(pos[b],n,logp,hyp,batch_idxs[0][b])
            dhyps[pos[b]-1] = hyp
            fh.write(out + '\n')
            fh.flush()
            if n+1 >= self.N:
              break

    if output != '-':
      fh.close()
    return [v for k,v in sorted(dhyps.items(), reverse=False)]


  def traverse_beam(self):
    bs =  self.z_src.shape[0]
    finals = [defaultdict() for i in range(bs)] #list with hyps reaching <eos> and overall score
    hyps = torch.ones([bs,1], dtype=int).to(self.device) * self.tgt_voc.idx_bos #[bs,lt=1]
    logP = torch.zeros([bs,1], dtype=torch.float32).to(self.device)     #[bs,lt=1]
    lp = self.batch_pre.shape[1] if self.prefix else 0 #max length of prefixes

    while True:
      #hyps is [I,lt] ; K is 1*K OR bs*K ; lt is the hyp length [1, 2, ..., max_size)
      I, lt = hyps.shape 

      if lt == 2:
        if self.model_type == 's_sc':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
        elif self.model_type == 's_s_scc_scc':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
          self.z_xtgt = self.z_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,lxt,ed] => [bs*K,lxt,ed]
          self.msk_xtgt = self.msk_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,1,lxt] => [bs*K,1,lxt]
        elif self.model_type == '2nmt_2c':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
          self.z_xtgt = self.z_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,lxt,ed] => [bs*K,lxt,ed]
          self.msk_xtgt = self.msk_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,1,lxt] => [bs*K,1,lxt]
          _, self.msk_tgt_cross = prepare_source_cross(hyps, self.tgt_voc.idx_pad, self.device)
        elif self.model_type == 'sxs_sc':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
          self.z_xtgt = self.z_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,lxt,ed] => [bs*K,lxt,ed]
          self.msk_xtgt = self.msk_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,1,lxt] => [bs*K,1,lxt]
        elif self.model_type == 'sxsc_sc':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
          self.z_xtgt = self.z_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,lxt,ed] => [bs*K,lxt,ed]
          self.msk_xtgt = self.msk_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,1,lxt] => [bs*K,1,lxt]
        elif self.model_type == 's_s_scc':
          self.z_src = self.z_src.repeat_interleave(repeats=self.K, dim=0) #[bs,ls,ed] => [bs*K,ls,ed]
          self.msk_src = self.msk_src.repeat_interleave(repeats=self.K, dim=0) #[bs,1,ls] => [bs*K,1,ls]
          self.z_xtgt = self.z_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,lxt,ed] => [bs*K,lxt,ed]
          self.msk_xtgt = self.msk_xtgt.repeat_interleave(repeats=self.K, dim=0) #[bs,1,lxt] => [bs*K,1,lxt]

      ##############
      ### DECODE ###
      ##############
      msk_tgt = (1 - torch.triu(torch.ones((1, lt, lt), device=self.device), diagonal=1)).bool()
      if self.model_type == 's_sc':
        y_next = self.model.decode(self.z_src, hyps, self.msk_src, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      elif self.model_type == 's_s_scc_scc':
        y_next = self.model.decode(self.z_src, self.z_xtgt, hyps, self.msk_src, self.msk_xtgt, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      elif self.model_type == '2nmt_2c':
        y_next = self.model.decode(self.z_src, self.z_xtgt, hyps, self.msk_src, self.msk_xtgt, msk_tgt=msk_tgt, msk_tgt_cross=msk_tgt_cross)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      elif self.model_type == 'sxs_sc':
        y_next = self.model.decode(self.z_src, self.z_xtgt, hyps, self.msk_src, self.msk_xtgt, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      elif self.model_type == 'sxsc_sc':
        y_next = self.model.decode(self.z_src, self.z_xtgt, hyps, self.msk_src, self.msk_xtgt, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      elif self.model_type == 's_s_scc':
        y_next = self.model.decode(self.z_src, self.z_xtgt, hyps, self.msk_src, self.msk_xtgt, msk_tgt=msk_tgt)[:,-1,:] #[I,lt,Vt] => [I,Vt]

      hyps, logP = self.expand(y_next, hyps, logP, bs) #both are [bs,1*Vt,lt] OR [bs,K*Vt,lt]
      
      if lt == self.max_size - 1: #last extension (force <eos> to appear in all hypotheses)
        logP = self.force_eos(logP) #[bs,1*Vt,lt] OR [bs,K*Vt,lt]

      elif self.prefix and lt < lp: #force decoding using prefix
        logP = self.force_prefix(hyps, logP, self.batch_pre[:,lt]) #[bs,1*Vt,lt] OR [bs,K*Vt,lt]

      hyps, logP = self.Kbest(hyps, logP) #both are [bs*K,lt]

      ##############
      ### FINALS ###
      ##############
      index_of_finals = (hyps[:,-1]==self.tgt_voc.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
      for i in index_of_finals:
        b = i//self.K
        if len(finals[b]) < self.K:
          hyp = ' '.join(map(str,hyps[i].tolist()))
          cst = sum(logP[i])
          if self.alpha:
            cst = cst / norm_length(hyps.shape[1],self.alpha)
          finals[b][hyp] = cst.item() # keep record of final hypothesis
          logP[i,-1] = -float('Inf') # force the hypothesis to disappear in next step

      if sum([len(d) for d in finals]) == bs*self.K:
        return finals


  def expand(self, y_next, hyps, logP, bs):
    #y_next is [I,Vt], I is either bs*1 OR bs*K
    #hyps is [I,lt]
    #logP is [I,lt]
    I, lt = hyps.shape
    next_logP = y_next.contiguous().view(-1,1) #[I*Vt,1]
    next_wrds = self.next_wrds.repeat_interleave(repeats=I, dim=0).view(-1,1) #[1,Vt] => [1*I,Vt] => [I*Vt,1]
    ##############
    ### EXPAND ###
    ##############
    hyps_expanded = hyps.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
    logP_expanded = logP.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
    ##############
    ### CONCAT ###
    ############## 
    hyps_expanded = torch.cat((hyps_expanded, next_wrds), dim=-1) #[I*Vt,lt+1]
    logP_expanded = torch.cat((logP_expanded, next_logP), dim=-1) #[I*Vt,lt+1]
    lt = hyps_expanded.shape[1] #new hyp length
    hyps_expanded = hyps_expanded.contiguous().view(bs,-1,lt) #[bs,1*Vt,lt] OR [bs,K*Vt,lt]
    logP_expanded = logP_expanded.contiguous().view(bs,-1,lt) #[bs,1*Vt,lt] OR [bs,K*Vt,lt]
    return hyps_expanded, logP_expanded


  def Kbest(self, hyps, logP):
    #hyps is [bs,n_times_Vt,lt] n is 1 or K
    #logP is [bs,n_times_Vt,lt]
    bs, n_times_Vt, lt = logP.shape
    sum_logP = torch.sum(logP,dim=2) #[bs,n_times_Vt] 
    _, kbest_inds = torch.topk(sum_logP, k=self.K, dim=1) #both are [bs,K] (finds the K-best of dimension 1) no need to norm-length since all have same length
    hyps = torch.stack([hyps[b][inds] for b,inds in enumerate(kbest_inds)], dim=0).contiguous().view(bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
    logP = torch.stack([logP[b][inds] for b,inds in enumerate(kbest_inds)], dim=0).contiguous().view(bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
    #self.print_beam(hyps, logP, bs, lt)
    return hyps, logP 


  def force_eos(self, logP):
    #logP is [bs, 1*Vt, lt] or [bs, K*Vt, lt]
    bs, n_times_Vt, lt = logP.shape
    logP = logP.contiguous().view(bs,-1,self.Vt,lt)

    #set -Inf to all last added tokens but idx_eos 
    all_but_eos = torch.cat( (torch.arange(0,self.tgt_voc.idx_eos), torch.arange(self.tgt_voc.idx_eos+1,self.Vt)) )
    logP[:,:,all_but_eos,-1] = float('-Inf') 

    logP = logP.contiguous().view(bs,n_times_Vt,lt)
    return logP


  def force_prefix(self, hyps, logP, pref):
    #hyps is [bs, 1*Vt, lt] or [bs, K*Vt, lt]
    #logP is [bs, 1*Vt, lt] or [bs, K*Vt, lt]
    #pref is [bs] (the prefix to be used for each bs)
    bs, n_times_Vt, lt = logP.shape
    logP = logP.contiguous().view(bs,-1,self.Vt,lt) #[bs,n,Vt,lt]

    for b in range(pref.shape[0]):
      idx_pref = pref[b].item()
      if idx_pref == self.tgt_voc.idx_eos: ### do not force if pref_idx is idx_eos
        continue
      elif idx_pref == self.tgt_voc.idx_pad: ### do not force if pref_idx is idx_pad
        continue
      all_Inf_but_pref = torch.cat( (torch.arange(0,idx_pref), torch.arange(idx_pref+1,self.Vt)) )
      logP[b,:,all_Inf_but_pref,-1] = float('-Inf') 

    logP = logP.contiguous().view(bs,n_times_Vt,lt)
    return logP


  def print_beam(self, hyps, logP, bs, lt):    
    hyps_bs_k = hyps.view(bs,self.K,lt)
    logP_bs_k = logP.view(bs,self.K,lt)
    for b in range(hyps_bs_k.shape[0]):
      for k in range(hyps_bs_k.shape[1]):
        logging.info('batch {} beam {}\tlogP={:.6f}\t{}'.format(b, k, sum(logP_bs_k[b,k]), ' '.join([self.tgt_voc[t] for t in hyps_bs_k[b,k].tolist()]) ))


  def format_hyp(self, p, n, c, tgt_idx, src_idx): 
    #p is the position in the input sentence
    #n is the position in the nbest 
    #c is the hypothesis overall cost (sum_logP_norm)
    #tgt_idx hypothesis (list of ints)
    #src_idx source (list of ints)
    while src_idx[-1] == self.src_voc.idx_pad: # eliminate <pad> tokens from src_idx
      src_idx = src_idx[:-1]

    hyp = []
    hyp.append(' '.join([self.tgt_voc[idx] for idx in tgt_idx[1:-1]])) ### output sentence (tokenized)

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
        out.append(' '.join([self.src_voc[idx] for idx in src_idx[1:-1]])) ### input sentence (tokenized)
      elif ch=='j':
        out.append(' '.join(map(str,src_idx))) ### input sentence (idxs)
      #########################
      ### target hypothesis ###
      #########################
      elif ch=='t':
        out.append(' '.join([self.tgt_voc[idx] for idx in tgt_idx[1:-1]])) ### output sentence (tokenized)
      elif ch=='i':
        out.append(' '.join(map(str,tgt_idx))) ### output sentence (idxs)

      else:
        logging.error('Invalid format option {} in {}'.format(ch,self.format))
        sys.exit()

    return '\t'.join(out), hyp










