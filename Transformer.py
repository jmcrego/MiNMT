# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
import torch
import math
from Options import Options
from Data import Vocab, Dataset, OpenNMTTokenizer
from Model import Encoder_Decoder, load_checkpoint_or_initialise, save_checkpoint, numparameters
from Optimizer import OptScheduler, LabelSmoothing
from Learning import Learning
import numpy as np
#import matplotlib.pyplot as plt

def plotPoints2d(X,Y,x=None,y=None,l=None,f=None):
  plt.figure(figsize=(15, 5))
  plt.plot(X,Y)
  if x is not None:
    plt.xlabel(x)
  if y is not None:
    plt.ylabel(y)
  if l is not None:
    plt.legend(l)
  if f is not None:
    plt.savefig(f)
  plt.show()

def plotMatrix2d(X,f=None):
  #print(X.shape)
  #print(X)
  plt.figure(figsize=(10, 10))
  if f is not None:
    plt.savefig(f)
  plt.imshow(X)

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  #a = torch.IntTensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]]])
  #print("a = {}\n{}".format(a.shape,a))
  #a = torch.index_select(a,dim=0,index=torch.tensor([1],dtype=torch.long)).squeeze()
  #print("a = {}\n{}".format(a.shape,a))
  #sys.exit()

  #lsrc = [5, 3, 2, 3, 3]
  #ltgt = [1, 3, 2, 5, 4]
  #print(lsrc)
  #print(ltgt)
  #print(np.lexsort((ltgt, lsrc)))
  #print(np.argsort(lsrc))
  #sys.exit()

  #src = np.asarray([np.asarray([1,2,3]), np.asarray([2,3,4]), np.asarray([1,2])])
  #src = [torch.tensor([1,2,3]), torch.tensor([2,3,4]), torch.tensor([1,2])]
  #src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
  #print(src)
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
  device = torch.device('cuda' if opts.cuda and torch.cuda.is_available() else 'cpu')
  logging.info('using device: {}'.format(device))
  model = Encoder_Decoder(on.n_layers, on.ff_dim, on.n_heads, on.emb_dim, on.qk_dim, on.v_dim, on.dropout, len(src_vocab), len(tgt_vocab), src_vocab.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device: {}'.format(', '.join([str(f) for f in numparameters(model)]), model.device))

  ################
  ### learning ###
  ################
  if od.train_set or (od.src_train and od.tgt_train):
    optim = torch.optim.Adam(model.parameters(), lr=oo.lr, betas=(oo.beta1, oo.beta2), eps=oo.eps)
    last_step, model, optim = load_checkpoint_or_initialise(opts.suffix, model, optim, device)
    optScheduler = OptScheduler(optim, on.emb_dim, oo.noam_scale, oo.noam_warmup, last_step)
    #plotPoints2d( [i for i in range(1,20000)],  [optScheduler.lrate(i) for i in range(1,20000)], '#Iter', 'LRate', ["dim={} scale={:.2f} warmup={}".format(on.emb_dim,oo.noam_scale,oo.noam_warmup)], 'kk.png')
    criter = LabelSmoothing(len(tgt_vocab), src_vocab.idx_pad, oo.label_smoothing).to(device)
    learning = Learning(model, optScheduler, criter, opts.suffix, ol)
    train = Dataset(src_vocab, tgt_vocab, src_token, tgt_token, od.src_train, od.tgt_train, od.shard_size, od.batch_size, od.train_set)
    if od.valid_set or (od.src_valid and od.tgt_valid):
      valid = Dataset(src_vocab, tgt_vocab, src_token, tgt_token, od.src_valid, od.tgt_valid, od.shard_size, od.batch_size, od.valid_set)
    else:
      valid = None
    learning.learn(train, valid, src_vocab.idx_pad, device)

  #################
  ### inference ###
  #################
  if od.test_set or od.src_test:
    _, model, _ = load_checkpoint(opts.suffix, model, None)
    inference = Inference(model, oi)
    test = Dataset(src_vocab, None, src_token, None, od.src_test, None, od.shard_size, od.batch_size, od.test_set)
    inference.translate(test)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
