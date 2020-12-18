# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

def prepare_input_src(batch_src, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] 
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  return src, msk_src #, msk_tgt

def prepare_input_tgt(seqs_tgt, idx_pad, device):
  tgt = seqs_tgt.to(device)
  msk_tgt = (tgt != idx_pad).unsqueeze(-2) & (1 - torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)).bool() #[bs,lt,lt]
  return tgt, msk_tgt

##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class BeamSearch():
  def __init__(self, model, tgt_vocab, beam_size, max_size, n_best, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = beam_size
    self.max_size = max_size
    self.n_best = n_best
    self.device = device

  def traverse(self, batch_src):
    src, msk_src = prepare_input_src(batch_src, self.tgt_vocab.idx_pad, self.device)
    K = self.beam_size #beam_size
    N = self.n_best    #nbest_size
    bs = src.shape[0]  #batch_size
    ls = src.shape[1]  #input sequence length
    Vt, ed = self.model.src_emb.weight.shape
    #src is [bs,ls]
    #msk_src is [bs,ls]
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    z_src = z_src.repeat_interleave(repeats=K, dim=0) #[bs*K,ls,ed] (repeats dimesion 0, K times)
    msk_src = msk_src.repeat_interleave(repeats=K, dim=0) #[bs*K,1,ls]
    #logging.info('z_src = {}'.format(z_src.shape))
    #logging.info('msk_src = {}'.format(msk_src.shape))

    ### initialize beam stack (it will always contain bs:batch_size and K:beam_size sentences) [initially sentences are '<bos>'] with logP=0.0
    beam_hyps = torch.ones([bs*K,1], dtype=int) * self.tgt_vocab['<bos>'] #(bs batches) (K beams) with one sentence each '<bos>' [bs*K,lt=1]
    beam_logP = torch.zeros([bs*K,1], dtype=torch.float32) #[bs*K, 1]

    for step in range(1,self.max_size+1):
      logging.info('Step {}'.format(step))
      logging.info('(1) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))
      tgt, msk_tgt = prepare_input_tgt(beam_hyps, self.tgt_vocab.idx_pad, self.device) #tgt is [bs*K, lt] msk_tgt is [bs*K, lt, lt]
      logging.info('tgt = {}'.format(tgt.shape))
      logging.info('msk_tgt = {}'.format(msk_tgt.shape))

      y = self.model.decode(z_src, tgt, msk_src, msk_tgt) #[bs*K, lt, Vt]
      y_next = y[:,-1] #[bs*K,Vt] #only interested on the last predicted token (next token)
      #logging.info('y_next = {}'.format(y_next.shape))
      next_logP, next_hyps = torch.topk(y_next, k=K, dim=1) #both are [bs*K, K]
      #logging.info('(1) next_hyps = {} next_logP = {}'.format(next_hyps.shape, next_logP.shape))
      beam_hyps, beam_logP = self.extend_beam_with_next(beam_hyps, beam_logP, next_hyps, next_logP, bs, K)

    sys.exit()

  def extend_beam_with_next(self, beam_hyps, beam_logP, next_hyps, next_logP, bs, K):
    next_logP = next_logP.contiguous().view(bs*K*K,1) #[bs*K*K,1]
    next_hyps = next_hyps.contiguous().view(bs*K*K,1) #[bs*K*K,1]
    logging.info('(2) next_hyps = {} next_logP = {}'.format(next_hyps.shape, next_logP.shape))

    beam_hyps = beam_hyps.repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,K*l] => [bs*K*K,l]
    beam_logP = beam_logP.repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,K] => [bs*K*K, 1]
    logging.info('(2) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### extend beam with new hyps (next)
    beam_hyps = torch.cat((beam_hyps, next_hyps), dim=-1) #[bs*K*K, lt]
    beam_logP += next_logP #[bs*K*K,1] + [bs*K*K,1] = [bs*K*K,1]
    logging.info('(3) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    beam_hyps = beam_hyps.contiguous().view(bs,K*K,-1) #[bs, K*K, lt]
    beam_logP = beam_logP.contiguous().view(bs,K*K)    #[bs, K*K]
    logging.info('(4) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    kbest_logP, kbest = torch.topk(beam_logP, k=K, dim=1) #both are [bs, K]
    logging.info('(1) kbest = {} kbest_logP = {}'.format(kbest.shape, kbest_logP.shape))

    beam_hyps = torch.stack([beam_hyps[t][inds] for t,inds in enumerate(kbest)], dim=0).contiguous().view(bs*K,-1)
    beam_logP = torch.gather(beam_logP, 1, kbest).contiguous().view(bs*K,1)
    logging.info('(5) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))
    for b in range(bs):
      logging.info(["{}:{}".format(w,self.tgt_vocab[w.item()]) for w in beam_hyps[b]])
    return beam_hyps, beam_logP



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

    b = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.max_size, self.n_best, device)
    for i_batch, (batch_src, _) in enumerate(testset):
      logging.debug('Translate #batch:{}'.format(i_batch))
      b.traverse(batch_src)

 









