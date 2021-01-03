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
  def __init__(self, bs, K, N, max_size, idx_bos, idx_eos, device):
    self.K = K #beam size
    self.bs = bs #batch size
    self.N = N #n-best
    self.max_size = max_size #max hyp length
    self.idx_bos = idx_bos
    self.idx_eos = idx_eos
    self.device = device
    self.beam_hyps = torch.ones([self.bs,1], dtype=int).to(self.device) * self.idx_bos #[bs,lt=1]
    self.beam_logP = torch.zeros([self.bs,1], dtype=torch.float32).to(self.device)     #[bs,lt=1]
    self.final = [defaultdict() for i in range(self.bs)]

  def done(self):
    if self.beam_hyps.shape[-1] >= self.max_size:
      logging.info('reached max_size={}'.format(self.max_size))
      return True
    for dhyps in self.final:
      if len(dhyps) < self.N:
        return False ### not enough final hyps for this beam
    return True

  def addfinal(self, b, h, c):
    self.final[b][' '.join(map(str,h))] = c

  def expand(self,y_next):
    #y_next is [B,lt=1] B is the number of hypotheses in y_next (either bs*1 or bs*K)

    #
    # keep the K-best options in y_next of each B
    #
    next_logP, next_hyps = torch.topk(y_next, k=self.K, dim=1) #both are [B,self.K]
    next_hyps = next_hyps.contiguous().view(-1,1) #[B*self.K,1]
    next_logP = next_logP.contiguous().view(-1,1) #[B*self.K,1]

    #
    # expand beam_hyps/beam_logP with next_hyps/next_logP
    #
    ### first expansion (beam_hyps/beam_logP are [self.bs,lt=1]) and next_hyps/next_logP are [self.bs*self.K,1]
    ### ulterior expansions (beam_hyps/beam_logP are [self.bs*self.K,lt>1]) and next_hyps/next_logP are [self.bs*self.K*self.K,1]
    #replicate each hyp in beam K times
    lt = self.beam_hyps.shape[1] #length of tgt hypotheses
    self.beam_hyps = self.beam_hyps.repeat_interleave(repeats=self.K, dim=0).contiguous().view(-1,lt) #[B,self.K*lt] => [B*self.K,lt=1]
    self.beam_logP = self.beam_logP.repeat_interleave(repeats=self.K, dim=0).contiguous().view(-1,lt) #[B,self.K*lt] => [B*self.K,lt=1]
    #logging.info('[replicate] beam_hyps = {} beam_logP = {}'.format(lt, self.beam_hyps.shape, self.beam_logP.shape))
    ### extend beam hyps with new word (next)
    self.beam_hyps = torch.cat((self.beam_hyps, next_hyps), dim=-1) #[B*self.K, lt+1]
    self.beam_logP = torch.cat((self.beam_logP, next_logP), dim=-1) #[B*self.K, lt+1]
    #logging.info('[cat] beam_hyps = {} beam_logP = {}'.format(lt, self.beam_hyps.shape, self.beam_logP.shape))

    lt = self.beam_hyps.shape[1]
    if self.K > 1 and self.beam_hyps.shape[0] == self.bs*self.K*self.K:
      #in the initial expansion we keep all generated hypotheses (bs*K) no need to do this
      #otherwise we must keep the K-best hyps among the bs*(K*K) available: reduce K*K into K
      self.beam_hyps = self.beam_hyps.contiguous().view(self.bs,self.K*self.K,lt)
      self.beam_logP = self.beam_logP.contiguous().view(self.bs,self.K*self.K,lt)
      kbest_logP, kbest_hyps = torch.topk(torch.sum(self.beam_logP,dim=2), k=self.K, dim=1) #both are [bs, K] (finds the K-best of dimension 1 (B*K)) no need to norm-length since all have same length
      #logging.info('kbest_hyps = {} kbest_logP = {}'.format(kbest_hyps.shape, kbest_logP.shape))
      self.beam_hyps = torch.stack([self.beam_hyps[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(self.bs*self.K,lt)
      self.beam_logP = torch.stack([self.beam_logP[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(self.bs*self.K,lt)
      #logging.info('[reduce] beam_hyps = {} beam_logP = {}'.format(lt, self.beam_hyps.shape, self.beam_logP.shape))

    ### find ending hypotheses in beam_hyps
    index_of_finals = (self.beam_hyps[:,-1]==self.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
    ### assign ending hypotheses -Inf logP and save them in self.final_hyps
    for i in index_of_finals.tolist():
      b = i//self.K #the beam it belongs
      h = self.beam_hyps[i].tolist() #[lt] hypothesis
      c = sum(self.beam_logP[i]) / norm_length(len(h),0.6) ### final cost of hypothesis normalized by length
      #print('i={} b={} c={:.5f} h: {}'.format(i,b,c,h))
      self.addfinal(b,h,c)
      self.beam_logP[i,-1] = -float('Inf') # this hyp wont be considered in next step

  def print(self, tgt_vocab):
    for b in range(self.bs):
      n = 0
      dicthyps = self.final[b]
      for hyp, cst in sorted(dicthyps.items(), key=lambda kv: kv[1], reverse=True):
        toks = [tgt_vocab[int(idx)] for idx in hyp.split(' ')]
        print('b={} {:.5f}\t{}'.format(b, cst, ' '.join(toks)))
        n += 1
        if n >= self.N:
          break

  def hyps(self):
    return self.beam_hyps 

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
    logging.info('Beam Search [init]: beam_size={} n_best={}'.format(self.beam_size,self.n_best))

  def traverse(self, batch_src):
    #Vt, ed = self.model.tgt_emb.weight.shape
    bs = len(batch_src) #batch_size
    K = self.beam_size
    #logging.info('Beam Search [traverse]: batch_size={}'.format(bs))
    #print(batch_src)
    ###
    ### encode the src sequence
    ###
    msk_src, z_src = encode_src(batch_src, self.model, self.tgt_vocab.idx_pad, self.device)
    #msk_src [bs,1,ls]
    #z_src [bs,ls,ed]
    ###
    ### decode step-by-step (produce one tgt token at each time step)
    ###
    beam = Beam(bs, self.beam_size, self.n_best, self.max_size, self.tgt_vocab.idx_bos, self.tgt_vocab.idx_eos, self.device)
    while not beam.done():
      y_next = self.model.decode(z_src, beam.hyps(), msk_src, msk_tgt=None)[:,-1,:] #[bs*K,lt,Vt] => [bs*K,Vt]
      beam.expand(y_next)
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
      for i_batch, (batch_src, _) in enumerate(testset):
        logging.debug('Translate #batch:{}'.format(i_batch))
        beam = beamsearch.traverse(batch_src)
        beam.print(self.tgt_vocab) 









