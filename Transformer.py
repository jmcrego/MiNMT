# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
import torch
from Options import Options
from Data import Vocab, Dataset, OpenNMTTokenizer
from Model import build_model, save_checkpoint, load_checkpoint_or_initialise
from Optimizer import OptScheduler, build_AdamOptimizer, LabelSmoothing
from Learning import Learning

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  #a = torch.IntTensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
  #print("a = {}\n{}".format(a.shape,a))
  #a = torch.index_select(a,dim=0,index=torch.tensor([1],dtype=torch.long)).squeeze()
  #print("a = {}\n{}".format(a.shape,a))
  #sys.exit()

  tic = time.time()
  opts = Options(sys.argv)
  ol = opts.learning
  on = opts.network
  oo = opts.optim
  od = opts.data
  oi = opts.inference

  src_token = OpenNMTTokenizer(od.src_token)
  tgt_token = OpenNMTTokenizer(od.tgt_token)
  src_vocab = Vocab(od.src_vocab)
  tgt_vocab = Vocab(od.tgt_vocab)
  assert src_vocab.idx_pad == tgt_vocab.idx_pad

  ################
  ### learning ###
  ################
  if od.train_set or (od.src_train and od.tgt_train):
    train = Dataset(src_vocab, tgt_vocab, src_token, tgt_token, od.src_train, od.tgt_train, od.shard_size, od.batch_size, od.train_set)
    if od.valid_set or (od.src_valid and od.tgt_valid):
      valid = Dataset(src_vocab, tgt_vocab, src_token, tgt_token, od.src_valid, od.tgt_valid, od.shard_size, od.batch_size, od.valid_set)
    else:
      valid = None
    model = build_model(on, len(src_vocab), len(tgt_vocab), src_vocab.idx_pad)
    optim = build_AdamOptimizer(model, oo.lr, oo.beta1, oo.beta2, oo.eps)
    last_step, model, optim = load_checkpoint_or_initialise(opts.suffix, model, optim)
    optScheduler = OptScheduler(optim, on.emb_dim, oo.noam_scale, oo.noam_warmup, last_step)
    criter = LabelSmoothing(len(tgt_vocab), src_vocab.idx_pad, oo.label_smoothing)
    learning = Learning(model, optScheduler, criter, opts.suffix, ol)
    learning.learn(train, valid)

  #################
  ### inference ###
  #################
  if od.test_set or od.src_test:
    test = Dataset(src_vocab, None, src_token, None, od.src_test, None, od.shard_size, od.batch_size, od.test_set)
    model = build_model(on, len(src_vocab), len(tgt_vocab), src_vocab.idx_pad)
    _, model, _ = load_checkpoint(opts.suffix, model, None)
    inference = Inference(model, oi)
    inference.translate(test)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
