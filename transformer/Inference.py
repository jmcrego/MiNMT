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
  def __init__(self, bs, K, N, max_size, alpha, tgt_vocab, device):
    self.bs = bs #batch size
    self.N = N #n-best
    self.K = K #beam size
    self.alpha = alpha
    self.max_size = max_size #max hyp length
    self.idx_bos = tgt_vocab.idx_bos
    self.idx_eos = tgt_vocab.idx_eos
    self.tgt_vocab = tgt_vocab
    self.Vt = len(tgt_vocab)
    self.device = device
    ### next are hypotheses and their corresponding costs (logP) maintained in beam
    self.hyps = torch.ones([bs,1], dtype=int).to(device) * self.idx_bos #[bs,lt=1]
    self.logP = torch.zeros([bs,1], dtype=torch.float32).to(device)     #[bs,lt=1]
    self.btch = torch.tensor([i for i in range(bs)]).view(-1,1).to(device) #[bs,1] this is to know to which example in batch belongs each hyp
    ### next are hyps reaching <eos>
    self.final = [defaultdict() for i in range(bs)] #list with hyps reaching <eos> and overall score
    self.debug = False
    self.force_eos = torch.ones(len(tgt_vocab)).to(device) * float('Inf') #[Vt]
    self.force_eos[self.idx_eos] = 1.0
    logging.debug('INITIAL')

  def done(self):
    for b in range(self.bs):
      if len(self.final[b]) < self.K:
        return False
    return True

  def advance(self,y_next):
    # y_next is [I,Vt] contains the proba ef expanding I hyps with each word in vocab
    # self.hyps/self.logP is [I,lt] self.btch is [I]
    assert y_next.shape[0] == self.hyps.shape[0] == self.logP.shape[0] == self.btch.shape[0]
    assert self.hyps.shape[1] == self.logP.shape[1]
    assert y_next.shape[1] == self.Vt
    I = self.hyps.shape[0] ### number of input hyps (to be expanded)
    lt = self.hyps.shape[1] #current length of tgt hypotheses

    if lt == self.max_size - 1: #last extension
      y_next[:,] *= self.force_eos #all words are assigned -Inf but <eos> which keeps its real logP

    ##################################################
    ### EXTEND hyps/logP with next_wrds/next_logP 
    ##################################################
    ###  prepare exapnsion (next words and their logP)
    next_logP, next_wrds = torch.topk(y_next, k=self.Vt, dim=1) #both are [I,Vt] (not really need to sort)
    next_wrds = next_wrds.contiguous().view(-1,1) #[I*Vt,1]
    next_logP = next_logP.contiguous().view(-1,1) #[I*Vt,1]
    #replicate each hyp in beam Vt times
    hyps_extended = self.hyps.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
    logP_extended = self.logP.repeat_interleave(repeats=self.Vt, dim=0) #[I,lt] => [I*Vt,lt]
    btch_extended = self.btch.repeat_interleave(repeats=self.Vt, dim=0) #[I,1]  => [I*Vt,1]
    # extend beam hyps with new word (next)
    hyps_extended = torch.cat((hyps_extended, next_wrds), dim=-1) #[I*Vt,lt+1]
    logP_extended = torch.cat((logP_extended, next_logP), dim=-1) #[I*Vt,lt+1]
    lt = hyps_extended.shape[1] #new hypothesis length

    #################################################
    ### REDUCE bs*(I*Vt) into bs*K to keep the K-best hyps of each hypothesis expanded
    #################################################

    hyps = []
    logP = []
    btch = []
    for b in range(self.bs):
      ### identify hyps of the same beam b
      inds_b = (btch_extended[:,0]==b).nonzero(as_tuple=False).squeeze(-1) ### hypotheses in btch_extended referred to the beam example b
      if len(inds_b) == 0: ### no alive hyps of this beam (already finished)
        continue
      hyps_b = hyps_extended[inds_b] #[I*Vt, lt]
      logP_b = logP_extended[inds_b] #[I*Vt, lt]
      btch_b = btch_extended[inds_b] #[I*Vt, 1]
      #keep the K-best of beam b
      sum_logP_b = torch.sum(logP_b,dim=1) #[I*Vt]
      _, inds_b_kbest = torch.topk(sum_logP_b, k=self.K, dim=0) #[K]
      hyps_b = hyps_b[inds_b_kbest] #[K, lt]
      logP_b = logP_b[inds_b_kbest] #[K, lt]
      btch_b = btch_b[inds_b_kbest] #[K, 1]
      ### identify final hypotheses
      index_of_not_finals = (hyps_b[:,-1]!=self.idx_eos).nonzero(as_tuple=False).squeeze(-1).tolist() #[n] n being the number of final hyps found
      for i in range(hyps_b.shape[0]):
        if i in index_of_not_finals:
          continue
        h = hyps_b[i].tolist() #[lt] hypothesis
        c = sum(logP_b[i]) / norm_length(len(h),self.alpha) ### final cost of hypothesis normalized by length
        self.final[b][' '.join(map(str,h))] = c ### save ending hyp into self.final
        _, sum_logP_norm, toks, logp = self.desc_hyp(btch_b[i], hyps_b[i], logP_b[i])
        logging.debug('[final]\ti={}\tb={}\t{:.5f}\t{}'.format(i,b,sum_logP_norm,toks))

      if len(index_of_not_finals) == 0: ### no alive hyps of this beam (already finished)
        continue
      ### eliminate finals from hyps_b/logP_b/btch_b
      hyps_b = hyps_b[index_of_not_finals]
      logP_b = logP_b[index_of_not_finals]
      btch_b = btch_b[index_of_not_finals]
      #accumulate non final hyps
      hyps.append(hyps_b)
      logP.append(logP_b)
      btch.append(btch_b)

    if len(hyps) > 0:
      self.hyps = torch.cat(hyps, dim=0)
      self.logP = torch.cat(logP, dim=0)
      self.btch = torch.cat(btch, dim=0)
      logging.debug('Hyps in BEAM = {} {}'.format(self.hyps.shape[0],self.btch.tolist()))
      self.print_beam('BEAM K={} lt={}'.format(self.K,lt))


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
    for i in range(self.btch.shape[0]):
      b, sum_logP_norm, toks_str, logp_str = self.desc_hyp(self.btch[i], self.hyps[i], self.logP[i])
      logging.debug('i={}\tb={}\t{:.5f}\t{}\t{}'.format(i, b, sum_logP_norm, toks_str, logp_str))

  def desc_hyp(self, btch_i, hyps_i, logP_i):
    b = btch_i.item()
    sum_logP_norm = sum(logP_i) / norm_length(hyps_i.shape[0],self.alpha)
    toks = ["{}".format(self.tgt_vocab[hyps_i[j].item()]) for j in range(hyps_i.shape[0])]
    logp = ["{:.4f}".format(logP_i[j].item()) for j in range(logP_i.shape[0])]
    return b, sum_logP_norm, ' '.join(toks), ' '.join(logp)

##############################################################################################################
### BeamSearch ###############################################################################################
##############################################################################################################
class BeamSearch():
  def __init__(self, model, tgt_vocab, beam_size, n_best, max_size, alpha, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.max_size = max_size
    self.device = device
    self.beam_size = beam_size
    self.n_best = n_best
    self.alpha = alpha

  def traverse(self, batch_src):
    # Vt, ed = self.model.tgt_emb.emb.weight.shape
    bs = len(batch_src)
    # K is beam_size
    ###
    ### encode the src sequence
    ###
    src, self.msk_src = prepare_source(batch_src, self.tgt_vocab.idx_pad, self.device) #src is [bs, ls] msk_src is [bs,1,ls]
    self.z_src = self.model.encode(src, self.msk_src) #[bs,ls,ed]
    ###
    ### decode step-by-step (produce one tgt token at each time step for each hyp in beam)
    ###
    beam = Beam(bs, self.beam_size, self.n_best, self.max_size, self.alpha, self.tgt_vocab, self.device)
    while not beam.done():
      z_src, msk_src = self.extend_source(beam.btch)
      y_next = self.model.decode(z_src, beam.hyps, msk_src, msk_tgt=None)[:,-1,:] #[I,lt,Vt] => [I,Vt]
      beam.advance(y_next)

    return beam

  def extend_source(self, btch):
    initialized = False
    for b in btch:
      if not initialized:
        z_src = self.z_src[b].detach().clone().to(self.device)
        msk_src = self.msk_src[b].detach().clone().to(self.device)
        initialized = True
      else:
        z_src = torch.cat((z_src, self.z_src[b]), dim=0)
        msk_src = torch.cat((msk_src, self.msk_src[b]),dim=0)
    return z_src, msk_src


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
      beamsearch = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.n_best, self.max_size, self.alpha, device)
      for pos, batch_src, _ in testset:
        beam = beamsearch.traverse(batch_src)
        logp, hyps = beam.get_hyps()
        assert len(pos) == len(batch_src) == len(logp) == len(hyps)
        for b in range(len(logp)): #each batch example
          for n in range(len(logp[b])): #each n-best
            hyp = hyps[b][n] #list of ints
            src = testset.get_input(pos[b])[1:-1] #list of strings (discard <bos> and <eos>)
            src_detok = self.src_token.detokenize(src) #string
            tgt = [self.tgt_vocab[idx] for idx in hyp[1:-1]] #list o strings (discard <bos> and <eos>)
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
              elif c=='u':
                out.append(' '.join(map(str,batch_src[b]))) ### input sentence (indexs)
              elif c=='h':
                out.append(' '.join(tgt))                   ### output sentence (tokenized)
              elif c=='v':
                out.append(' '.join(map(str,hyp)))          ### output sentence (indexs)
              elif c=='H':
                out.append(tgt_detok)                       ### output sentence (detokenized)
              else:
                logging.error('invalid format option {} in {}'.format(c,self.format))
                sys.exit()
            print('\t'.join(out), flush=True)








