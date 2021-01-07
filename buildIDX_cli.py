# -*- coding: utf-8 -*-

import sys
import os
import time
import pickle
import logging
import numpy as np
from transformer.Data import Dataset
from transformer.Vocab import Vocab
from transformer.ONMTTokenizer import ONMTTokenizer
#import matplotlib.pyplot as plt

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.debug('Created Logger level={}'.format(loglevel))
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.debug('Created Logger level={} file={}'.format(loglevel, logfile))

class Options():

  def __init__(self, argv):
    self.fsrc = None
    self.ftgt = None
    self.fset = None
    self.src_vocab = None
    self.tgt_vocab = None
    self.src_token = None
    self.tgt_token = None
    self.shard_size = 100000

    self.prog = argv.pop(0)
    log_file = None
    log_level = 'info'

    while len(argv):
      tok = argv.pop(0)
      if tok=="-h":
        self.usage()
      elif tok=="-src" and len(argv):
        self.fsrc = argv.pop(0)
      elif tok=="-tgt" and len(argv):
        self.ftgt = argv.pop(0)

      elif tok=="-set" and len(argv):
        self.fset = argv.pop(0)

      elif tok=="-src_vocab" and len(argv):
        self.src_vocab = argv.pop(0)
      elif tok=="-tgt_vocab" and len(argv):
        self.tgt_vocab = argv.pop(0)

      elif tok=="-src_token" and len(argv):
        self.src_token = argv.pop(0)
      elif tok=="-tgt_token" and len(argv):
        self.tgt_token = argv.pop(0)

      elif tok=="-shard_size" and len(argv):
        self.shard_size = int(argv.pop(0))

      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

      else:
        sys.stderr.write('error: unparsed {} option\n'.format(tok))
        self.usage()

    create_logger(log_file,log_level)

    if self.fsrc is None:
      logging.error("Missing -src option")
      self.usage()


    if self.src_vocab is None:
      logging.error("Missing -src_vocab option")
      self.usage()

    if self.src_token is None:
      logging.error("Missing -src_token option")
      self.usage()


  def usage(self):
    sys.stderr.write('''usage: {} -src FILE -tgt FILE -set FILE -src_vocab FILE -tgt_vocab FILE -src_token FILE -tgt_token FILE [-h] [-log_level LEVEL] [-log_file FILE]
   -shard_size   INT : size of shards (100000)
   -log_file    FILE : log file  (stderr)
   -log_level STRING : log level [debug, info, warning, critical, error] (info)
   -h                : this help
This program:
- numberizes text files using given Vocabs and Tokenizers 
- randomizes all examples
- splits examples in a set of shards
- examples in each shard are sorted by src and tgt lengths
- saves the set of shards in a binary file
'''.format(self.prog))
    sys.exit()



######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)
  src_vocab = Vocab(ONMTTokenizer(fyaml=o.src_token), file=o.src_vocab)
  if o.tgt_vocab:
    tgt_vocab = Vocab(ONMTTokenizer(fyaml=o.tgt_token), file=o.tgt_vocab)
    assert src_vocab.idx_pad == tgt_vocab.idx_pad
  else:
    tgt_vocab = None

  d = Dataset(src_vocab, tgt_vocab)
  d.numberize(o.fsrc, o.ftgt)
  d.split_in_shards(o.shard_size)
  d.dump_shards(o.fset)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
