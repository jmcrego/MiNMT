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
    self.dnet = None
    self.src_voc = None
    self.tgt_voc = None

    self.n = {}
    self.n['emb_dim'] = 512
    self.n['qk_dim'] = 64
    self.n['v_dim'] = 64
    self.n['ff_dim'] = 2048
    self.n['n_heads'] = 8
    self.n['n_layers'] = 6
    self.n['dropout'] = 0.1
    self.n['share_embeddings'] = False

    self.n['weight_decay'] = 0.0
    self.n['beta1'] = 0.9
    self.n['beta2'] = 0.998
    self.n['eps'] = 1e-9

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
        self.n['emb_dim'] = int(argv.pop(0))
      elif tok=="-qk_dim" and len(argv):
        self.n['qk_dim'] = int(argv.pop(0))
      elif tok=="-v_dim" and len(argv):
        self.n['v_dim'] = int(argv.pop(0))
      elif tok=="-ff_dim" and len(argv):
        self.n['ff_dim'] = int(argv.pop(0))
      elif tok=="-n_heads" and len(argv):
        self.n['n_heads'] = int(argv.pop(0))
      elif tok=="-n_layers" and len(argv):
        self.n['n_layers'] = int(argv.pop(0))
      elif tok=="-dropout" and len(argv):
        self.n['dropout'] = float(argv.pop(0))
      elif tok=="-share_embeddings":
        self.n['share_embeddings'] = True

      elif tok=='-weight_decay':
        self.n['weight_decay'] = float(argv.pop(0))
      elif tok=='-beta1':
        self.n['beta1'] = float(argv.pop(0))
      elif tok=='-beta2':
        self.n['beta2'] = float(argv.pop(0))
      elif tok=='-eps':
        self.n['eps'] = float(argv.pop(0))

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
    sys.stderr.write('''usage: {} -dnet DIR -src_voc FILE -tgt_voc FILE [Options]
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

   -weight_decay  FLOAT : weight decay for Adam optimizer ({})
   -beta1         FLOAT : beta1 for Adam optimizer ({})
   -beta2         FLOAT : beta2 for Adam optimizer ({})
   -eps           FLOAT : epsilon for Adam optimizer ({})

   -log_file       FILE : log file  (stderr)
   -log_level       STR : log level [debug, info, warning, critical, error] (info)
   -h                   : this help
'''.format(self.prog, self.n['emb_dim'], self.n['qk_dim'], self.n['v_dim'], self.n['ff_dim'], self.n['n_heads'], self.n['n_layers'], self.n['dropout'], self.n['share_embeddings'], self.n['weight_decay'], self.n['beta1'], self.n['beta2'], self.n['eps']))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)
  write_dnet(o) ### saves network, src_voc, tgt_voc
  src_voc = Vocab(o.src_voc)
  tgt_voc = Vocab(o.tgt_voc)
  device = torch.device('cpu')
  model = Encoder_Decoder(o.n['n_layers'], o.n['ff_dim'], o.n['n_heads'], o.n['emb_dim'], o.n['qk_dim'], o.n['v_dim'], o.n['dropout'], o.n['share_embeddings'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))
  model = initialise(model)
  optim = torch.optim.Adam(model.parameters(), weight_decay=o.n['weight_decay'], betas=(o.n['beta1'], o.n['beta2']), eps=o.n['eps']) 
  save_checkpoint(o.dnet+ '/network', model, optim, 0, 0) ### saves model checkpoint and optimizer
  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
