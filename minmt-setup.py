#!/usr/bin/env python3

import sys
import os
import time
import logging
import torch
from tools.Tools import create_logger, write_dnet
from transformer.Dataset import Vocab
from transformer.Model import Encoder_Decoder, save_checkpoint, numparameters
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
    self.net = {} ### contains all network parameters
    self.net['emb_dim'] = 512
    self.net['qk_dim'] = 64
    self.net['v_dim'] = 64
    self.net['ff_dim'] = 2048
    self.net['n_heads'] = 8
    self.net['n_layers'] = 6
    self.net['dropout'] = 0.1
    self.net['share_embeddings'] = False
    self.net['weight_decay'] = 0.0
    self.net['beta1'] = 0.9
    self.net['beta2'] = 0.998
    self.net['eps'] = 1e-9
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
        self.net['emb_dim'] = int(argv.pop(0))
      elif tok=="-qk_dim" and len(argv):
        self.net['qk_dim'] = int(argv.pop(0))
      elif tok=="-v_dim" and len(argv):
        self.net['v_dim'] = int(argv.pop(0))
      elif tok=="-ff_dim" and len(argv):
        self.net['ff_dim'] = int(argv.pop(0))
      elif tok=="-n_heads" and len(argv):
        self.net['n_heads'] = int(argv.pop(0))
      elif tok=="-n_layers" and len(argv):
        self.net['n_layers'] = int(argv.pop(0))
      elif tok=="-dropout" and len(argv):
        self.net['dropout'] = float(argv.pop(0))
      elif tok=="-share_embeddings":
        self.net['share_embeddings'] = True
      elif tok=='-weight_decay' and len(argv):
        self.net['weight_decay'] = float(argv.pop(0))
      elif tok=='-beta1' and len(argv):
        self.net['beta1'] = float(argv.pop(0))
      elif tok=='-beta2' and len(argv):
        self.net['beta2'] = float(argv.pop(0))
      elif tok=='-eps' and len(argv):
        self.net['eps'] = float(argv.pop(0))

      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

      else:
        self.usage('Unrecognized {} option'.format(tok))

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

  def usage(self, messg=None):
    if messg is not None:
      sys.stderr.write(messg + '\n')
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
'''.format(self.prog, self.net['emb_dim'], self.net['qk_dim'], self.net['v_dim'], self.net['ff_dim'], self.net['n_heads'], self.net['n_layers'], self.net['dropout'], self.net['share_embeddings'], self.net['weight_decay'], self.net['beta1'], self.net['beta2'], self.net['eps']))
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
  model = Encoder_Decoder(o.net['n_layers'], o.net['ff_dim'], o.net['n_heads'], o.net['emb_dim'], o.net['qk_dim'], o.net['v_dim'], o.net['dropout'], o.net['share_embeddings'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))
  for p in model.parameters():
    if p.dim() > 1:
      torch.nn.init.xavier_uniform_(p)
  logging.info('[network initialised]')
  optim = torch.optim.Adam(model.parameters(), weight_decay=o.net['weight_decay'], betas=(o.net['beta1'], o.net['beta2']), eps=o.net['eps']) 
  save_checkpoint(o.dnet+ '/network', model, optim, 0, 0) ### saves model checkpoint and optimizer
  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
