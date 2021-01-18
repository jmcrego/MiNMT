# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import torch
from transformer.Model import prepare_source

def norm_length(l, alpha):
  if alpha == 0.0:
    return 1.0
  return (5+l)**alpha / (5+1)**alpha


##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class Beam():
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
    self.debug = False
    if self.debug:
      self.print_beam('INITIAL')

  def done(self):
    if self.hyps.shape[-1] >= self.max_size: ### stop if already prduced max_size tokens in hyps
      return True    
    for dhyps in self.final: ### stop if all beams already produced K (beam_size) final hypotheses
      if len(dhyps) < self.K:
        return False 
    return True ### do not stop

  def advance(self,y_next):
    I = self.hyps.shape[0] ### number of input hyps (to expand)
    assert y_next.shape[0] == self.hyps.shape[0]
    # I is either:
    # - bs*1 (beam just created containing <eos> of each example in batch)
    # - bs*K (batch_size * beam_size)
    lt = self.hyps.shape[1] #current length of tgt hypotheses
    # self.hyps is [I,lt]
    # y_next is [I,Vt] contains the proba ef expanding I hyps with each word in vocab
    assert y_next.shape[0] == self.bs or y_next.shape[0] == self.bs*self.K
    assert y_next.shape[1] == len(self.tgt_vocab)
    #This function extends hypotheses in self.hyps with one word keeping the k-best [bs*K,lt+1]

    # we keep the K-best choices for each hypothesis in y_next
    next_logP, next_wrds = torch.topk(y_next, k=self.K, dim=1) #both are [I,self.K]
    next_wrds = next_wrds.contiguous().view(-1,1) #[I*self.K,1]
    next_logP = next_logP.contiguous().view(-1,1) #[I*self.K,1]
    logging.debug('***** EXTEND with {}-best next_wrds: {}'.format(self.K, [self.tgt_vocab[idx] for idx in next_wrds.view(-1).tolist()]))

    ###
    ### EXPAND hyps/logP with next_wrds/next_logP
    ###
    ### first expand (hyps/logP are [self.bs,lt=1]) and next_wrds/next_logP are [self.bs*self.K,1]
    ### ulterior expansions (hyps/logP are [self.bs*self.K,lt>1]) and next_wrds/next_logP are [self.bs*self.K*self.K,1]
    #replicate each hyp in beam K times
    self.hyps = self.hyps.repeat_interleave(repeats=self.K, dim=0) #[I,lt] => [I*self.K,lt]
    self.logP = self.logP.repeat_interleave(repeats=self.K, dim=0) #[I,lt] => [I*self.K,lt]
    ### extend beam hyps with new word (next)
    self.hyps = torch.cat((self.hyps, next_wrds), dim=-1) #[I*self.K,lt+1]
    self.logP = torch.cat((self.logP, next_logP), dim=-1) #[I*self.K,lt+1]
    lt = self.hyps.shape[1]
    self.print_beam('EXPAND K={}'.format(self.K))

    ###
    ### REDUCE bs*(K*K) into bs*K to keep the K-best hyps of each example in batch (not done in initial expansion since only bs*1*K hyps nor when K=1)
    ###
    if self.hyps.shape[0] > self.bs*self.K:
      self.hyps = self.hyps.contiguous().view(self.bs,-1,lt) #[bs,K*K,lt] or [bs,1*K,lt] (first expand)
      self.logP = self.logP.contiguous().view(self.bs,-1,lt) #[bs,K*K,lt] or [bs,1*K,lt] (first expand)
      kbest_logP, kbest_hyps = torch.topk(torch.sum(self.logP,dim=2), k=self.K, dim=1) #both are [bs, K] (finds the K-best of dimension 1 (I*K)) no need to norm-length since all have same length
      self.hyps = torch.stack([self.hyps[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(self.bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
      self.logP = torch.stack([self.logP[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(self.bs*self.K,lt) #[bs,K,lt] => [bs*K,lt]
      self.print_beam('REDUCE K={}'.format(self.K))

    ###
    ### Final hypotheses
    ###
    index_of_finals = (self.hyps[:,-1]==self.idx_eos).nonzero(as_tuple=False).squeeze(-1) #[n] n being the number of final hyps found
    for i in index_of_finals.tolist():
      b = i//self.K #the batch example where it belongs
      h = self.hyps[i].tolist() #[lt] hypothesis
      c = sum(self.logP[i]) / norm_length(len(h),self.alpha) ### final cost of hypothesis normalized by length
      self.final[b][' '.join(map(str,h))] = c ### save ending hyp into self.final (discard <bos> and <eos>)
      self.logP[i,-1] = -float('Inf') # assign ending hypotheses -Inf so wont remain in beam the next time step
      logging.debug('[final i={}]'.format(i))

  def get_hyps(self):
    hyps = []
    logp = []
    for b in range(self.bs):
      hyps.append([])
      logp.append([])
      dicthyps = self.final[b]
      for hyp, sum_logP_norm in sorted(dicthyps.items(), key=lambda kv: kv[1], reverse=True):
        hyps[-1].append(list(map(int, hyp.split(' ')))) 
        logp[-1].append(sum_logP_norm.item())
        if len(hyps[-1]) >= self.N:
          break
    return logp, hyps

  def print_beam(self, tag):
    logging.debug('[{}] hyps.size={}'.format(tag, self.hyps.shape[1]))    
    for i in range(self.hyps.shape[0]):
      sum_logP_norm = sum(self.logP[i]) / norm_length(self.hyps.shape[1],self.alpha)
      if False:
        toks = ["{:.4f}:{}".format(self.logP[i,j].item(),self.tgt_vocab[self.hyps[i,j].item()]) for j in range(len(self.hyps[i]))]
        logging.debug('i={}\t{:.5f}\t{}'.format(i,sum_logP_norm,' '.join(toks)))
      else:
        toks1 = ["{}".format(self.tgt_vocab[self.hyps[i,j].item()]) for j in range(len(self.hyps[i]))]
        toks2 = ["{:.4f}".format(self.logP[i,j].item()) for j in range(len(self.hyps[i]))]
        logging.debug('i={} b={}\t{:.5f}\t{}\t{}'.format(i,int(i/self.K),sum_logP_norm,' '.join(toks1),' '.join(toks2)))
    

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
    #Vt, ed = self.model.tgt_emb.emb.weight.shape
    bs = len(batch_src) #batch_size
    K = self.beam_size
    ###
    ### encode the src sequence
    ###
    src, msk_src = prepare_source(batch_src, self.tgt_vocab.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    ###
    ### decode step-by-step (produce one tgt token at each time step for each hyp in beam)
    ###
    beam = Beam(bs, self.beam_size, self.n_best, self.max_size, self.tgt_vocab, self.device)
    while not beam.done():
      y_next = self.model.decode(z_src, beam.hyps, msk_src, msk_tgt=None)[:,-1,:] #[bs*K,lt,Vt] => [bs*K,Vt]
      beam.advance(y_next)
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
  def __init__(self, model, tgt_vocab, src_token, tgt_token, oi): 
    super(Inference, self).__init__()
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.src_token = src_token
    self.tgt_token = tgt_token
    self.beam_size = oi.beam_size
    self.max_size = oi.max_size
    self.n_best = oi.n_best
    self.alpha = oi.alpha
    self.format = oi.format

  def translate(self, testset, device):
    logging.info('Running: inference')

    with torch.no_grad():
      self.model.eval()
      beamsearch = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.n_best, self.max_size, device)
      for pos, batch_src, _ in testset:
        beam = beamsearch.traverse(batch_src)
        logp, hyps = beam.get_hyps()
        assert len(pos) == len(batch_src) == len(logp) == len(hyps)
        for b in range(len(logp)):
          for n in range(len(logp[b])):
            hyp = hyps[b][n] #list of ints
            src = testset.get_input(pos[b])[1:-1] #list of strings
            src_detok = self.src_token.detokenize(src) #string
            tgt = [self.tgt_vocab[idx] for idx in hyp[1:-1]] #list
            tgt_detok = self.tgt_token.detokenize(tgt) #string
            out = []
            for c in self.format:
              if c=='i':
                out.append("{}".format(pos[b]+1))           ### position in input file
              elif c=='n':
                out.append("{}".format(n+1))                ### n-best order
              elif c=='c':
                out.append("{:.6f}".format(logp[b][n]))     ### cost (logP)
              elif c=='s':
                out.append(' '.join(src))                   ### input sentence (tokenized)
              elif c=='S':
                out.append(src_detok)                       ### input sentence (detokenized)
                #out.append(' '.join(map(str,batch_src[b])))### input sentence (indexs)
              elif c=='h':
                out.append(' '.join(tgt))                   ### output sentence (tokenized)
                #out.append(' '.join(map(str,hyp)))         ### output sentence (indexs)
              elif c=='H':
                out.append(tgt_detok)                       ### output sentence (detokenized)
              else:
                logging.error('invalid format option {} in {}'.format(c,self.format))
                sys.exit()
            print('\t'.join(out), flush=True)








