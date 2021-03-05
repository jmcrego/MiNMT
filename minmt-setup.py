#!/usr/bin/env python3

import sys
import os
import time
import logging
import torch
from tools.Tools import create_logger, write_dnet
from transformer.Dataset import Vocab
from transformer.Model import Encoder_Decoder, save_checkpoint, numparameters, initialise
import numpy as np

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)

    self.src_voc = None
    self.tgt_voc = None

    self.emb_dim = 512
    self.qk_dim = 64
    self.v_dim = 64
    self.ff_dim = 2048
    self.n_heads = 8
    self.n_layers = 6
    self.dropout = 0.1
    self.share_embeddings = False
    self.dnet = None

    self.weight_decay = 0.0
    self.beta1 = 0.9
    self.beta2 = 0.998
    self.eps = 1e-9

    log_file = 'stderr'
    log_level = 'info'

    while len(argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()

      elif tok=="-dnet" and len(argv):
        self.dnet = argv.pop(0)
      elif tok=="-src_voc" and len(argv):
        self.src_voc = argv.pop(0)
      elif tok=="-tgt_voc" and len(argv):
        self.tgt_voc = argv.pop(0)

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

      elif tok=='-weight_decay':
        self.weight_decay= float(argv.pop(0))
      elif tok=='-beta1':
        self.beta1 = float(argv.pop(0))
      elif tok=='-beta2':
        self.beta2 = float(argv.pop(0))
      elif tok=='-eps':
        self.eps = float(argv.pop(0))

      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    create_logger(log_file,log_level)
    if self.dnet is None:
      logging.error('missing -dnet option')
      self.usage()
    if self.src_voc is None:
      logging.error('missing -src_voc option')
      self.usage()
    if self.tgt_voc is None:
      logging.error('missing -tgt_voc option')
      self.usage()

  def usage(self):
    sys.stderr.write('''usage: {} -dnet DIR [Options]
   -dnet            DIR : network ouput directory [must not exist]
   -src_voc        FILE : source vocabulary file
   -tgt_voc        FILE : target vocabulary file

   -emb_dim         INT : model embedding dimension ({})
   -qk_dim          INT : query/key embedding dimension ({})
   -v_dim           INT : value embedding dimension ({})
   -ff_dim          INT : feed-forward inner layer dimension ({})
   -n_heads         INT : number of attention heads ({})
   -n_layers        INT : number of encoder layers ({})
   -dropout       FLOAT : dropout value ({})
   -share_embeddings    : share source/target embeddings ({})

   -weight_decay    FLOAT : Adam optimizer weight decay ({})
   -beta1           FLOAT : beta1 for Adam optimizer ({})
   -beta2           FLOAT : beta2 for Adam optimizer ({})
   -eps             FLOAT : epsilon for Adam optimizer ({})

   -log_file    FILE : log file  (stderr)
   -log_level    STR : log level [debug, info, warning, critical, error] (info)
   -h                : this help
'''.format(self.prog, self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers, self.dropout, self.share_embeddings, self.weight_decay, self.beta1, self.beta2, self.eps))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)
  src_voc = Vocab(o.src_voc)
  tgt_voc = Vocab(o.tgt_voc)
  device = torch.device('cpu')
  model = Encoder_Decoder(o.n_layers, o.ff_dim, o.n_heads, o.emb_dim, o.qk_dim, o.v_dim, o.dropout, o.share_embeddings, len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))
  model = initialise(model)
  optim = torch.optim.Adam(model.parameters(), weight_decay=o.weight_decay, betas=(o.beta1, o.beta2), eps=o.eps) 
  write_dnet(o) ### writes network, src_voc, tgt_voc
  save_checkpoint(o.dnet+ '/network', model, optim, 0, 0)
  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
