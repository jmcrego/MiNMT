# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import torch

def encode_src(batch_src, model, idx_pad, device):
  src = [torch.tensor(seq) for seq in batch_src] #[bs, ls]
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device) #src is [bs,ls]
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  z_src = model.encode(src, msk_src) #[bs,ls,ed]
  return msk_src, z_src

def norm_length(l, alpha):
  return (5+l)**alpha / (5+1)**alpha

##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class Beam():
  #def __init__(self, bs, K, N, max_size, idx_bos, idx_eos, device):
  def __init__(self, bs, K, N, max_size, tgt_vocab, device):
    self.bs = bs #batch size
    self.N = N #n-best
    self.K = K #beam size
    self.alpha = 0.7
    self.max_size = max_size #max hyp length
    self.idx_bos = tgt_vocab.idx_bos
    self.idx_eos = tgt_vocab.idx_eos
    self.tgt_vocab = tgt_vocab
    self.device = device
    ### next are hypotheses and their corresponding cost (logP) maintained in beam
    self.hyps = torch.ones([self.bs,1], dtype=int).to(self.device) * self.idx_bos #[bs,lt=1]
    self.logP = torch.zeros([self.bs,1], dtype=torch.float32).to(self.device)     #[bs,lt=1]
    ### next are hyps reaching <eos>
    self.final = [defaultdict() for i in range(self.bs)] #list with hyps reaching <eos> and overall score
    self.print_beam('INITIAL')

  def done(self):
    ### stop if already prduced max_size tokens in hyps
    if self.hyps.shape[-1] >= self.max_size: 
      return True
    ### stop if all beams already produced K (beam_size) final hypotheses
    for dhyps in self.final: 
      if len(dhyps) < self.K:
        return False 
    ### stop if lowest scored hyp is lower than corresponding best hyp-threshold 

    ### do not stop
    return True

  def expand_breath(self,y_next):
    #y_next is [B,Vt] B is the number of hypotheses in y_next. Either:
    # bs*1 (beam just created containing <eos> of each batch_size hypothesis)
    # bs*K (batch_size * beam_size)
    # y_next contains Vt probs for each hypothesis in hyps
    assert y_next.shape[0] == self.bs or y_next.shape[0] == self.bs*self.K
    assert y_next.shape[0] == self.hyps.shape[0]
    assert y_next.shape[1] == len(self.tgt_vocab)
    #The output of this function is always batch_size * beam_size (bs*K)
    I = y_next.shape[0] ### number of input hyps (input to expand)
    O = self.bs*self.K  ### number of output hyps (output to expand)
    lt = self.hyps.shape[1] #length of tgt hypotheses

    # we keep the K-best choices for each hypothesis in y_next
    next_logP, next_hyps = torch.topk(y_next, k=self.K, dim=1) #both are [I,self.K]
    next_hyps = next_hyps.contiguous().view(-1,1) #[I*self.K,1]
    next_logP = next_logP.contiguous().view(-1,1) #[I*self.K,1]
    print('----------> {}-best next_hyps: {}'.format(self.K, [self.tgt_vocab[idx] for idx in next_hyps.view(-1).tolist()]))

    ###
    ### EXPAND hyps/logP with next_hyps/next_logP
    ###
    ### first expand (hyps/logP are [self.bs,lt=1]) and next_hyps/next_logP are [self.bs*self.K,1]
    ### ulterior expansions (hyps/logP are [self.bs*self.K,lt>1]) and next_hyps/next_logP are [self.bs*self.K*self.K,1]
    #replicate each hyp in beam K times
    self.hyps = self.hyps.repeat_interleave(repeats=self.K, dim=0) #[I,lt] => [I*self.K,lt]
    self.logP = self.logP.repeat_interleave(repeats=self.K, dim=0) #[I,lt] => [I*self.K,lt]
    ### extend beam hyps with new word (next)
    self.hyps = torch.cat((self.hyps, next_hyps), dim=-1) #[I*self.K,lt+1]
    self.logP = torch.cat((self.logP, next_logP), dim=-1) #[I*self.K,lt+1]
    lt = self.hyps.shape[1]
    self.print_beam('EXPAND K={}'.format(self.K))
    #logging.info('hyps = {} logP = {}'.format(self.hyps.shape, self.logP.shape))

    ###
    ### REDUCE bs*(K*K) into bs*K to keep the K-best hyps of each example in batch (not done in initial expansion since only bs*1*K hyps nor when K=1)
    ###
    if self.hyps.shape[0] > self.bs*self.K:
      self.hyps = self.hyps.contiguous().view(self.bs,-1,lt) #[bs,K*K,lt] or [bs,1*K,lt] (first expand)
      self.logP = self.logP.contiguous().view(self.bs,-1,lt) #[bs,K*K,lt] or [bs,1*K,lt] (first expand)
      kbest_logP, kbest_hyps = torch.topk(torch.sum(self.logP,dim=2), k=self.K, dim=1) #both are [bs, K] (finds the K-best of dimension 1 (I*K)) no need to norm-length since all have same length
      self.hyps = torch.stack([self.hyps[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0) #[bs,K,lt]
      self.logP = torch.stack([self.logP[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0) #[bs,K,lt]
      self.hyps = self.hyps.contiguous().view(self.bs*self.K,lt) #[bs*K,lt]
      self.logP = self.logP.contiguous().view(self.bs*self.K,lt) #[bs*K,lt]
      self.print_beam('REDUCE K={}'.format(self.K))

    ###
    ### check ending hypotheses
    ###
    index_of_finals = (self.hyps[:,-1]==self.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
    for i in index_of_finals.tolist():
      b = i//self.K #the batch example where it belongs
      h = self.hyps[i].tolist() #[lt] hypothesis
      c = sum(self.logP[i]) / norm_length(len(h),self.alpha) ### final cost of hypothesis normalized by length
      self.final[b][' '.join(map(str,h))] = c ### save ending hyp into self.final
      self.logP[i,-1] = -float('Inf') # assign ending hypotheses -Inf so wont remain in beam the next time step
      print('[final hyp i={}]'.format(i))

  def print_nbest(self, pos):
    for b in range(self.bs):
      n = 0
      dicthyps = self.final[b]
      for hyp, cst in sorted(dicthyps.items(), key=lambda kv: kv[1], reverse=True):
        toks = [self.tgt_vocab[int(idx)] for idx in hyp.split(' ')]
        print('{} {} {:.5f}\t{}'.format(pos[b], n+1, cst, ' '.join(toks)))
        n += 1
        if n >= self.N:
          break

  def print_beam(self, tag):
    lt = self.hyps.shape[1]
    print('[{}] hyps.size={}'.format(tag, self.hyps.shape[1]))    
    for i in range(self.hyps.shape[0]):
      toks = ["{:.4f}:{}".format(self.logP[i,j].item(),self.tgt_vocab[self.hyps[i,j].item()]) for j in range(len(self.hyps[i]))]
      sum_logP = sum(self.logP[i]) / norm_length(lt,self.alpha)
      print('i={}\t{:.5f}\t{}'.format(i,sum_logP,' '.join(toks)))


##############################################################################################################
### BeamSearch ###############################################################################################
##############################################################################################################
class BeamSearch():
  def __init__(self, model, tgt_vocab, beam_size, n_best, max_size, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.max_size = max_size
    self.device = device
    self.beam_size = beam_size
    self.n_best = n_best
    #logging.info('Beam Search [init]: beam_size={} n_best={}'.format(self.beam_size,self.n_best))

  def traverse(self, batch_src):
    #Vt, ed = self.model.tgt_emb.weight.shape
    bs = len(batch_src) #batch_size
    K = self.beam_size
    ###
    ### encode the src sequence
    ###
    msk_src, z_src = encode_src(batch_src, self.model, self.tgt_vocab.idx_pad, self.device)
    #msk_src [bs,1,ls]
    #z_src [bs,ls,ed]
    ###
    ### decode step-by-step (produce one tgt token at each time step for each hyp in beam)
    ###
    beam = Beam(bs, self.beam_size, self.n_best, self.max_size, self.tgt_vocab, self.device)
    while not beam.done():
      y_next = self.model.decode(z_src, beam.hyps, msk_src, msk_tgt=None)[:,-1,:] #[bs*K,lt,Vt] => [bs*K,Vt]
      beam.expand_breath(y_next)
      #beam.print_beam(self.tgt_vocab)
      ### from now on i decode bs*K hyps (i need z_src/msk_src to be the same shape)
      if self.beam_size > 1 and msk_src.shape[0] == bs:
        msk_src = msk_src.repeat_interleave(repeats=K, dim=0) #[bs*K,1,ls] 
        z_src = z_src.repeat_interleave(repeats=K, dim=0) #[bs*K,ls,ed] 

    return beam


##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, tgt_vocab, oi): 
    super(Inference, self).__init__()
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = oi.beam_size
    self.max_size = oi.max_size
    self.n_best = oi.n_best

  def translate(self, testset, device):
    logging.info('Running: inference')

    with torch.no_grad():
      self.model.eval()
      beamsearch = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.n_best, self.max_size, device)
      for pos, batch_src, _ in testset:
        beam = beamsearch.traverse(batch_src)
        beam.print_nbest(pos) 









