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
    Vt, ed = self.model.tgt_emb.weight.shape
    #src is [bs,ls]
    #msk_src is [bs,ls]
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    z_src = z_src.repeat_interleave(repeats=K, dim=0) #[bs*K,ls,ed] (repeats dimesion 0, K times)
    msk_src = msk_src.repeat_interleave(repeats=K, dim=0) #[bs*K,1,ls]
    #logging.info('z_src = {}'.format(z_src.shape))
    #logging.info('msk_src = {}'.format(msk_src.shape))

    ### initialize beam stack (it will always contain bs:batch_size and K:beam_size sentences) [initially sentences are '<bos>'] with logP=0.0
    beam_hyps = torch.ones([bs*K,1], dtype=int) * self.tgt_vocab['<bos>'] #(bs batches) (K beams) with one sentence each '<bos>' [bs*K,lt=1]
    beam_logP = torch.zeros([bs*K], dtype=torch.float32) #[bs*K]

    for lt in range(1,self.max_size+1):
      logging.info('Step {}'.format(lt))
      tgt, msk_tgt = prepare_input_tgt(beam_hyps, self.tgt_vocab.idx_pad, self.device) #tgt is [bs*K, lt] msk_tgt is [bs*K, lt, lt]
      y = self.model.decode(z_src, tgt, msk_src, msk_tgt) #[bs*K, lt, Vt]
      y_next = y[:,-1,:] #[bs*K,Vt] #only interested on the last predicted token (next token)
      next_logP, next_hyps = torch.topk(y_next, k=K, dim=1) #both are [bs*K,K]
      beam_hyps, beam_logP = self.extend_beam_with_next(beam_hyps, beam_logP, next_hyps, next_logP, bs, K) #[bs*K,lt] and [bs*K]

      for h in range(len(beam_hyps)):
        sys.stdout.write('hyp[{}]:'.format(h))
        for idx in beam_hyps[h]:
          sys.stdout.write(' {}:{}'.format(idx.item(),self.tgt_vocab[idx.item()]))
        print()

    sys.exit()

  def extend_beam_with_next(self, beam_hyps, beam_logP, next_hyps, next_logP, bs, K):
    lt = beam_hyps.shape[1]
    assert bs*K == beam_hyps.shape[0]

    next_hyps = next_hyps.contiguous().view(bs*K*K,1) #[bs*K*K,1]
    next_logP = next_logP.contiguous().view(bs*K*K,1) #[bs*K*K,1]
    #logging.info('(2a) next_hyps = {} next_logP = {}'.format(next_hyps.shape, next_logP.shape))
    #print(next_hyps)
    #print(next_logP)

    beam_hyps = beam_hyps.repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,K*l] => [bs*K*K,l]
    beam_logP = beam_logP.contiguous().view(bs*K,1).repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,1] => [bs*K,K] => [bs*K*K, 1]
    #logging.info('(2a) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))
    #print(beam_hyps)
    #print(beam_logP)

    ### extend beam with new hyps (next)
    beam_hyps = torch.cat((beam_hyps, next_hyps), dim=-1) #[bs*K*K, lt]
    beam_logP += next_logP #[bs*K*K,1] + [bs*K*K,1] = [bs*K*K,1]
    #logging.info('(2b) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))
    #print(beam_hyps)
    #print(beam_logP)

    lt = beam_hyps.shape[1]
    beam_hyps = beam_hyps.contiguous().view(bs,K*K,lt) #[bs, K*K, lt]
    beam_logP = beam_logP.contiguous().view(bs,K*K)    #[bs, K*K]
    #logging.info('(2c) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### keep the K-best of each batch
    kbest_logP, kbest_hyps = torch.topk(beam_logP, k=K, dim=1) #both are [bs, K]
    #logging.info('(2) kbest_hyps = {} kbest_logP = {}'.format(kbest_hyps.shape, kbest_logP.shape))
    #print(kbest_hyps)
    #print(kbest_logP)

    if False:
      new_beam_hyps = torch.zeros([bs,K,lt], dtype=int)
      new_beam_logP = torch.zeros([bs,K], dtype=torch.float32)
      for b in range(bs):
        for k in range(K):
          myk = kbest_hyps[b,k]
          new_beam_logP[b,k] = beam_logP[b,myk].item()
          for l in range(lt):
            new_beam_hyps[b,k,l] = beam_hyps[b,myk,l].item()
  
      new_beam_hyps = new_beam_hyps.view(bs*K,lt)
      new_beam_logP = new_beam_logP.view(bs*K)

    new_beam_hyps = torch.stack([beam_hyps[t][inds] for t,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(bs*K,-1)
    new_beam_logP = torch.gather(beam_logP, 1, kbest_hyps).contiguous().view(bs*K,1)

    return new_beam_hyps, new_beam_logP



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

 









