# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
import torch
import math
from Options import Options
from Data import Dataset
from Vocab import Vocab
from ONMTTokenizer import ONMTTokenizer
from Model import Encoder_Decoder, load_checkpoint_or_initialise, save_checkpoint, load_checkpoint, numparameters
from Optimizer import OptScheduler, LabelSmoothing
from Learning import Learning
from Inference import Inference
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



  #a = torch.IntTensor([1,2,1,2,1,2])
  #a = a.view([3,2])
  #print(a)

  #b = torch.IntTensor([3,3,3])
  #b = b.view(3,1)
  #print(b)

  #a = torch.cat((a, b), dim=1)
  #print(a)

  #sys.exit()

  #K=3
  #a = a.view([-1])
  #a = a.repeat_interleave(repeats=K, dim=0)
  #print(a.shape)  
  #print(a)
  #sys.exit()
  #print("a = {}\n{}".format(a.shape,a))
  #a = torch.index_select(a,dim=0,index=torch.tensor([1],dtype=torch.long)).squeeze()
  #print("a = {}\n{}".format(a.shape,a))
  #a = a.unsqueeze(1).repeat_interleave(repeats=2, dim=1)
  #a = a.repeat_interleave(repeats=2, dim=0)
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
    #plotPoints2d( [i for i in range(1,20000)],  [optScheduler.lrate(i) for i in range(1,20000)], '#Iter', 'LRate', ["dim={} scale={:.2f} warmup={}".format(on.emb_dim,oo.noam_scale,oo.noam_warmup)], 'kk.png')
    criter = LabelSmoothing(len(tgt_vocab), src_vocab.idx_pad, oo.label_smoothing).to(device)
    learning = Learning(model, optScheduler, criter, opts.suffix, src_vocab.idx_pad, ol)
    valid = load_dataset(src_vocab, tgt_vocab, od.valid_set, od.src_valid, od.tgt_valid, od.shard_size, ol.max_length, ol.batch_size, ol.batch_type)
    train = load_dataset(src_vocab, tgt_vocab, od.train_set, od.src_train, od.tgt_train, od.shard_size, ol.max_length, ol.batch_size, ol.batch_type)
    learning.learn(train, valid, device)

  #################
  ### inference ###
  #################
  if od.test_set or od.src_test:
    model = load_checkpoint(opts.suffix, model, device)
    test = load_dataset(src_vocab, None, od.test_set, od.src_test, None, od.shard_size, ol.max_length, ol.batch_size, ol.batch_type)
    inference = Inference(model, tgt_vocab, oi)
    inference.translate(test, device)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
