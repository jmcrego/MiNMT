# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
import logging
from tools.Preprocessor import SentencePiece, Space
from tools.Tools import create_logger
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
      elif tok=="-preprocessor" and len(argv):
        self.preprocessor = argv.pop(0)
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
   -preprocessor STRING : sentencepiece/space ({})

   -log_file    FILE : log file  (stderr)
   -log_level    STR : log level [debug, info, warning, critical, error] (info)
   -h                : this help
'''.format(self.prog, self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers, self.dropout, self.share_embeddings, self.preprocessor))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  opts = Options(sys.argv)

  if os.path.exists(opts.dnet):
    logging.error('cannot create network directory: {}'.format(opts.dnet))
    sys.exit()
  if not os.path.isfile(opts.src_pre):
    logging.error('cannot find source preprocessor file: {}'.format(opts.src_pre))
    sys.exit()
  if not os.path.isfile(opts.tgt_pre):
    logging.error('cannot find target preprocessor file: {}'.format(opts.tgt_pre))
    sys.exit()

  os.mkdir(opts.dnet)
  logging.info('created network directory: {}'.format(opts.dnet))
  with open(opts.dnet+'/network', 'w') as f:
    f.write('emb_dim: {}\n'.format(opts.emb_dim))
    f.write('qk_dim: {}\n'.format(opts.qk_dim))
    f.write('v_dim: {}\n'.format(opts.v_dim))
    f.write('ff_dim: {}\n'.format(opts.ff_dim))
    f.write('n_heads: {}\n'.format(opts.n_heads))
    f.write('n_layers: {}\n'.format(opts.n_layers))
    f.write('dropout: {}\n'.format(opts.dropout))
    f.write('share_embeddings: {}\n'.format(opts.share_embeddings))
    f.write('preprocessor: {}\n'.format(opts.preprocessor))

  shutil.copy(opts.src_pre, opts.dnet+'/src_pre')
  logging.info('copied source preprocessor {} into {}/src_pre'.format(opts.src_pre, opts.dnet))

  shutil.copy(opts.tgt_pre, opts.dnet+'/tgt_pre')
  logging.info('copied target preprocessor {} into {}/tgt_pre'.format(opts.tgt_pre, opts.dnet))

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
