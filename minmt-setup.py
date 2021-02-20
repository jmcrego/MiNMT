# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
import logging
from tools.Preprocessor import SentencePiece, Space
from tools.Tools import create_logger, write_dnet
import numpy as np

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)
    self.emb_dim = 512
    self.qk_dim = 64
    self.v_dim = 64
    self.ff_dim = 2048
    self.n_heads = 8
    self.n_layers = 6
    self.dropout = 0.1
    self.share_embeddings = False
    self.preprocessor = 'space'
    self.dnet = None
    self.src_pre = None
    self.tgt_pre = None

    log_file = 'stderr'
    log_level = 'info'

    while len(argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()
      elif tok=="-emb_dim" and len(argv):
        self.emb_dim = int(argv.pop(0))
      elif tok=="-qk_dim" and len(argv):
        self.qk_dim = int(argv.pop(0))
      elif tok=="-v_dim" and len(argv):
        self.v_dim = int(argv.pop(0))
      elif tok=="-ff_dim" and len(argv):
        self.ff_dim = int(argv.pop(0))
      elif tok=="-n_heads" and len(argv):
        self.n_heads = int(argv.pop(0))
      elif tok=="-n_layers" and len(argv):
        self.n_layers = int(argv.pop(0))
      elif tok=="-dropout" and len(argv):
        self.dropout = float(argv.pop(0))
      elif tok=="-share_embeddings":
        self.share_embeddings = True
      elif tok=="-dnet" and len(argv):
        self.dnet = argv.pop(0)
      elif tok=="-src_pre" and len(argv):
        self.src_pre = argv.pop(0)
      elif tok=="-tgt_pre" and len(argv):
        self.tgt_pre = argv.pop(0)
      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    create_logger(log_file,log_level)
    if self.dnet is None:
      logging.error('missing -dnet option')
      self.usage()
    if self.src_pre is None:
      logging.error('missing -src_pre option')
      self.usage()
    if self.tgt_pre is None:
      logging.error('missing -tgt_pre option')
      self.usage()

  def usage(self):
    sys.stderr.write('''usage: {} -dnet FILE [Options]
   -dnet            DIR : network ouput directory [must not exist]
   -src_pre        FILE : source preprocessor file
   -tgt_pre        FILE : target preprocessor file

   -emb_dim         INT : model embedding dimension ({})
   -qk_dim          INT : query/key embedding dimension ({})
   -v_dim           INT : value embedding dimension ({})
   -ff_dim          INT : feed-forward inner layer dimension ({})
   -n_heads         INT : number of attention heads ({})
   -n_layers        INT : number of encoder layers ({})
   -dropout       FLOAT : dropout value ({})
   -share_embeddings    : share source/target embeddings ({})

   -log_file    FILE : log file  (stderr)
   -log_level    STR : log level [debug, info, warning, critical, error] (info)
   -h                : this help
'''.format(self.prog, self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers, self.dropout, self.share_embeddings))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)
  write_dnet(o)
  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
