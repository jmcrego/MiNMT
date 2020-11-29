# -*- coding: utf-8 -*-

import sys
import time
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

  train = Dataset(opts.data.src_vocab, opts.data.tgt_vocab, opts.data.src_token, opts.data.tgt_token, opts.data.src_train, opts.data.tgt_train, opts.data.shard_size, opts.data.batch_size)
  sys.exit()

  model = build_model(opts, opts.data.src_vocab, opts.data.src_vocab)

  if opts.run == 'learning':
    logging.info('Running: learning')

  elif opts.run == 'inference':
    logging.info('Running: inference')

  else:
    logging.warning('Nothing to run')

  toc = time.time()
  logging.info('Done ({:.3f} seconds)'.format(toc-tic))











    
