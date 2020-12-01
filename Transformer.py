# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
from Options import Options
from Data import Vocab, Dataset
from Model import build_model

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  opts = Options(sys.argv)
  tic = time.time()

  if opts.data.train_set or (opts.data.src_train and opts.data.tgt_train):
    train = Dataset(opts.data.src_vocab, opts.data.tgt_vocab, opts.data.src_token, opts.data.tgt_token, opts.data.src_train, opts.data.tgt_train, opts.data.shard_size, opts.data.batch_size, opts.data.train_set)
  if opts.data.valid_set or (opts.data.src_valid and opts.data.tgt_valid):
    valid = Dataset(opts.data.src_vocab, opts.data.tgt_vocab, opts.data.src_token, opts.data.tgt_token, opts.data.src_valid, opts.data.tgt_valid, 0, opts.data.batch_size, opts.data.valid_set)
  if opts.data.test_set or opts.data.src_test:
    test = Dataset(opts.data.src_vocab, None, opts.data.src_token, None, opts.data.src_test, None, 0, opts.data.batch_size, opts.data.test_set)

  model = build_model(opts)

  if opts.run == 'learning':
    logging.info('Running: learning')

  elif opts.run == 'inference':
    logging.info('Running: inference')

  else:
    logging.warning('Nothing to run')

  toc = time.time()
  logging.info('Done ({:.3f} seconds)'.format(toc-tic))











    
