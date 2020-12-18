# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

def prepare_input_src(batch_src, max_length, idx_pad, device):
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
  def __init__(self, model, tgt_vocab, beam_size, max_length, n_best, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = beam_size
    self.max_length = max_length
    self.n_best = n_best
    self.device = device

  def traverse(self, batch_src):
    src, msk_src = prepare_input_src(batch_src, self.max_length, self.tgt_vocab.idx_pad, self.device)
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

    ### initialize history
    history = torch.ones([bs*K,1], dtype=int) * self.tgt_vocab['<bos>'] #(bs batches) (K beams) with one sentence each '<bos>' [bs*K,lt=1]
    history_logP = torch.zeros([bs*K,1], dtype=torch.float32) #[bs*K, 1]

    for step in range(self.max_length):
      logging.info('step {}'.format(step))
      logging.info('(1) history = {} history_logP = {}'.format(history.shape, history_logP.shape))
      tgt, msk_tgt = prepare_input_tgt(history, self.tgt_vocab.idx_pad, self.device) #tgt is [bs*K, lt] msk_tgt is [bs*K, lt, lt]
      #logging.info('z_src = {}'.format(z_src.shape))
      #logging.info('msk_src = {}'.format(msk_src.shape))
      #logging.info('tgt = {}'.format(tgt.shape))
      #logging.info('msk_tgt = {}'.format(msk_tgt.shape))
      y = self.model.decode(z_src, tgt, msk_src, msk_tgt) #[bs*K, lt, Vt]
      #only interested on the last predicted token 
      y = y[:,-1] #[bs*K,Vt]
      logging.info('y = {}'.format(y.shape))

      kbest_logP, kbest = torch.topk(y, k=K, dim=1) #both are [bs*K, K]
      logging.info('(1) kbest = {} kbest_logP = {}'.format(kbest.shape, kbest_logP.shape))
      kbest_logP = kbest_logP.contiguous().view(bs*K*K,1)
      kbest = kbest.contiguous().view(bs*K*K,1)
      logging.info('(2) kbest = {} kbest_logP = {}'.format(kbest.shape, kbest_logP.shape))

      history_logP = history_logP.repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,K] => [bs*K*K, 1]
      history = history.repeat_interleave(repeats=K, dim=1).contiguous().view(bs*K*K,-1) #[bs*K,K*l] => [bs*K*K,l]
      logging.info('(2) history = {} history_logP = {}'.format(history.shape, history_logP.shape))

      history = torch.cat((history, kbest), dim=-1) #[bs*K*K, lt]
      history_logP += kbest_logP #[bs*K*K,1] + [bs*K*K,1] = [bs*K*K,1]
      logging.info('(3) history = {} history_logP = {}'.format(history.shape, history_logP.shape))

      history = history.view(bs,K*K,-1) #[bs, K*K, lt]
      history_logP = history_logP.contiguous().view(bs,K*K) #[bs, K*K]
      logging.info('(4) history = {} history_logP = {}'.format(history.shape, history_logP.shape))

      kbest_logP, kbest = torch.topk(history_logP, k=K, dim=1) #both are [bs, K]
      logging.info('(3) kbest = {} kbest_logP = {}'.format(kbest.shape, kbest_logP.shape))

      history = torch.stack([history[t][inds] for t,inds in enumerate(kbest)], dim=0).contiguous().view(bs*K,-1)
      history_logP = torch.gather(history_logP, 1, kbest).contiguous().view(bs*K,1)
      logging.info('(5) history = {} history_logP = {}'.format(history.shape, history_logP.shape))
      logging.info([self.tgt_vocab[w] for w in history[0]])
    sys.exit()


  def update_history(self, history, history_logprb, kbest_ind, kbest_logprb):
    #history is #[bs*K,1]
    #history_logprb is #[bs*K,1]
    logging.info('history = {}'.format(history.shape))
    logging.info('history_logprb = {}'.format(history_logprb.shape))
    logging.info('kbest_ind = {}'.format(kbest_ind.shape))
    logging.info('kbest_logprb = {}'.format(kbest_logprb.shape))
    sys.exit()

##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, tgt_vocab, oi): 
    super(Inference, self).__init__()
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = oi.beam_size
    self.max_length = oi.max_length
    self.n_best = oi.n_best

  def translate(self, testset, device):
    logging.info('Running: inference')

    b = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.max_length, self.n_best, device)
    for i_batch, (batch_src, _) in enumerate(testset):
      logging.debug('Translate #batch:{}'.format(i_batch))
      b.traverse(batch_src)

 









