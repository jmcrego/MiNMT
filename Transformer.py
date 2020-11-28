# -*- coding: utf-8 -*-

import sys
import time
import logging
from Options import Options
from Data import Vocab
from Model import build_model

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  opts = Options(sys.argv)
  tic = time.time()

  src_vocab = Vocab(opts.data.src_vocab)
  tgt_vocab = Vocab(opts.data.tgt_vocab)
  model = build_model(opts,src_vocab,tgt_vocab)

  if opts.run == 'learning':
    logging.info('Running: learning')

  elif opts.run == 'inference':
    logging.info('Running: inference')

  else:
    logging.warning('Nothing to run')

  toc = time.time()
  logging.info('Done ({:.3f} seconds)'.format(toc-tic))











    
