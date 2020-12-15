# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

def prepare_input_src(batch_src, max_length, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] #as is
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  return src, msk_src #, msk_tgt

def prepare_input_tgt(seqs_tgt, idx_pad, device):
#  tgt = [torch.tensor(seq) for seq in seqs_tgt] 
#  tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=idx_pad).to(device) 
  tgt = torch.tensor(seqs_tgt).to(device)
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
    bs = src.shape[0]  #batch_size
    #src is [bs,ls]
    #msk_src is [bs,ls]
    logging.info('src = {}'.format(src.shape)) 
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    logging.info('z_src = {}'.format(z_src.shape))

    history = torch.ones([K,bs,1], dtype=int) * self.tgt_vocab['<bos>'] 

    for step in range(self.max_length):
      logging.info('history = {}'.format(history))
      for k in range(K):
        tgt, msk_tgt = prepare_input_tgt(history[k], self.tgt_vocab.idx_pad, self.device)
        #tgt is [bs, lt]
        #msk_tgt is [bs, lt, lt]
        logging.info("tgt:{}".format(tgt.shape)) #[bs, lt]
        logging.info("msk_tgt:{}".format(msk_tgt.shape)) #[bs, lt]
        y = self.model.decode(z_src, tgt, msk_src, msk_tgt) #[bs, lt, Vt]
        logging.info("y:{}".format(y.shape)) #[bs, lt, Vt]
        val_kbest, ind_kbest = torch.topk(y, K)
        logging.info("step:{} ind_kbest:{}".format(step,ind_kbest.shape)) #[bs, lt, K]
        print([self.tgt_vocab[ind] for ind.item() in ind_kbest[0][-1]])
        #concat result with corresponding hystory and assign score

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

 









