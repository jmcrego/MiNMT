# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

def prepare_input(batch_src, max_length, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] #as is
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  msk_tgt = (1 - torch.triu(torch.ones((1, max_length, max_length), device=device), diagonal=1)).bool() #[bs,lt,lt]
  return src, msk_src, msk_tgt

##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class Beam():
  def __init__(self, model, idx_pad, beam_size, max_length, n_best, device):
    assert idx_pad == model.idx_pad
    self.model = model
    self.idx_pad = idx_pad
    self.beam_size = beam_size
    self.max_length = max_length
    self.n_best = n_best
    self.device = device

  def output(self):
    pass

  def traverse(self, batch_src):
    src, msk_src, msk_tgt = prepare_input(batch_src, self.max_length, self.idx_pad, self.device)
    z_src = self.model.encode(src, msk_src)
    logging.info('z_src = {}'.format(z_src.shape))
#    z_tgt = previous histories (embedded tgt words)
#    tgt = <bos>
#    for step in range(self.max_length):
#      z_tgt = self.model.decode_step(self, z_src, z_tgt, tgt, msk_src, msk_tgt): 
#      tgt = self.generator(z_tgt)


##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, idx_pad, oi): 
    super(Inference, self).__init__()
    self.model = model
    self.idx_pad = idx_pad
    self.beam_size = oi.beam_size
    self.max_length = oi.max_length
    self.n_best = oi.n_best

  def translate(self, testset, device):
    logging.info('Running: inference')

    b = Beam(self.model, self.idx_pad, self.beam_size, self.max_length, self.n_best, device)
    for i_batch, (batch_src, _) in enumerate(testset):
      logging.debug('Translate #batch:{}'.format(i_batch))
      b.traverse(batch_src)
      b.output()

 









