# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
import torch
import math
from transformer.Options import Options
from transformer.Data import Dataset
from transformer.Vocab import Vocab
from transformer.ONMTTokenizer import ONMTTokenizer
from transformer.Model import Encoder_Decoder, load_checkpoint_or_initialise, save_checkpoint, load_checkpoint, numparameters
from transformer.Optimizer import OptScheduler, LabelSmoothing, NLLLoss
from transformer.Learning import Learning
from transformer.Inference import Inference
import numpy as np

def load_dataset(src_vocab, tgt_vocab, fset, fsrc, ftgt, shard_size, max_length, batch_size, batch_type):
  d = Dataset(src_vocab, tgt_vocab)
  if fset is not None and os.path.exists(fset):
    d.load_shards(fset)
    d.split_in_batches(max_length, batch_size, batch_type)
    return d

  d.numberize(fsrc, ftgt)
  d.split_in_shards(shard_size)
  if fset is not None:
    d.dump_shards(fset)
  d.split_in_batches(max_length, batch_size, batch_type)
  return d

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  opts = Options(sys.argv)
  ol = opts.learning
  on = opts.network
  oo = opts.optim
  od = opts.data
  oi = opts.inference

  src_vocab = Vocab(ONMTTokenizer(fyaml=od.src_token), file=od.src_vocab)
  tgt_vocab = Vocab(ONMTTokenizer(fyaml=od.tgt_token), file=od.tgt_vocab)
  assert src_vocab.idx_pad == tgt_vocab.idx_pad

  device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')
  model = Encoder_Decoder(on.n_layers, on.ff_dim, on.n_heads, on.emb_dim, on.qk_dim, on.v_dim, on.dropout, len(src_vocab), len(tgt_vocab), src_vocab.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))

  ################
  ### learning ###
  ################
  if od.train_set or (od.src_train and od.tgt_train):
    optim = torch.optim.Adam(model.parameters(), lr=oo.lr, betas=(oo.beta1, oo.beta2), eps=oo.eps)
    last_step, model, optim = load_checkpoint_or_initialise(opts.suffix, model, optim, device)
    optScheduler = OptScheduler(optim, on.emb_dim, oo.noam_scale, oo.noam_warmup, last_step)
    #criter = LabelSmoothing(len(tgt_vocab), src_vocab.idx_pad, oo.label_smoothing).to(device)
    criter = NLLLoss(len(tgt_vocab), src_vocab.idx_pad).to(device)
    learning = Learning(model, optScheduler, criter, opts.suffix, src_vocab.idx_pad, ol)
    valid = load_dataset(src_vocab, tgt_vocab, od.valid_set, od.src_valid, od.tgt_valid, od.shard_size, od.max_length, od.batch_size, od.batch_type)
    train = load_dataset(src_vocab, tgt_vocab, od.train_set, od.src_train, od.tgt_train, od.shard_size, od.max_length, od.batch_size, od.batch_type)
    learning.learn(train, valid, device)

  #################
  ### inference ###
  #################
  if od.test_set or od.src_test:
    model = load_checkpoint(opts.suffix, model, device)
    test = load_dataset(src_vocab, None, od.test_set, od.src_test, None, od.shard_size, od.max_length, od.batch_size, od.batch_type)
    inference = Inference(model, tgt_vocab, oi)
    inference.translate(test, device)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
