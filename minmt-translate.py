# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import torch
import yaml
from transformer.Dataset import Dataset
from transformer.Model import Encoder_Decoder, load_checkpoint_or_initialise, save_checkpoint, load_checkpoint, numparameters
from transformer.Inference import Inference
from tools.Preprocessor import SentencePiece, Space
from tools.Tools import create_logger

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)
    self.dnet = None
    self.input = None
    self.output = '-'
    self.beam_size = 4
    self.n_best = 1
    self.max_size = 250
    self.alpha = 0.0
    self.format = 'pT'
    self.shard_size = 0
    self.max_length = 0
    self.batch_size = 30
    self.batch_type = 'sentences'    
    self.cuda = False
    log_file = 'stderr'
    log_level = 'info'

    while len(argv):
      tok = argv.pop(0)

      if tok=="-h":
        self.usage()

      elif tok=='-dnet' and len(argv):
        self.dnet = argv.pop(0)
      elif tok=='-beam_size' and len(argv):
        self.beam_size = int(argv.pop(0))
      elif tok=='-n_best' and len(argv):
        self.n_best = int(argv.pop(0))
      elif tok=='-max_size' and len(argv):
        self.max_size = int(argv.pop(0))
      elif tok=='-alpha' and len(argv):
        self.alpha = float(argv.pop(0))
      elif tok=='-format' and len(argv):
        self.format = argv.pop(0)
      elif tok=='-i' and len(argv):
        self.input = argv.pop(0)
      elif tok=='-o' and len(argv):
        self.output = argv.pop(0)
      elif tok=='-shard_size' and len(argv):
        self.shard_size = int(argv.pop(0))
      elif tok=='-max_length' and len(argv):
        self.max_length = int(argv.pop(0))
      elif tok=='-batch_size' and len(argv):
        self.batch_size = int(argv.pop(0))
      elif tok=='-batch_type' and len(argv):
        self.batch_type = argv.pop(0)

      elif tok=="-cuda":
        self.cuda = True
      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    if self.dnet is None:
      self.usage('missing -dnet option')
    if self.input is None:
      self.usage('missing -i option')
    create_logger(log_file,log_level)
    logging.info("Options = {}".format(self.__dict__))


  def usage(self, messg=None):
    if messg is not None:
      sys.stderr.write(messg + '\n')
    sys.stderr.write('''usage: {} -dnet DIR -i FILE [Options]
   -dnet          DIR : network directory [must exist]
   -i            FILE : input file to translate
   -o            FILE : output file ({})

   [Inference]
   -beam_size     INT : size of beam ({})
   -n_best        INT : return n-best translation hypotheses ({})
   -max_size      INT : max hypothesis size ({})
   -alpha       FLOAT : hypothesis length-normalization parameter ({}) [use 0.0 for unnormalized otherwise (5+len)**alpha / (5+1)**alpha]
   -format     STRING : format of output lines (default {})
                          [p] index in test set
                          [n] rank in n-best
                          [c] global hypothesis cost
                          [s] input sentence
                          [S] input sentence (detokenised)
                          [j] input sentence ids
                          [t] target hypothesis
                          [T] target hypothesis (detokenised)
                          [i] target hypothesis ids

   [Data]
   -shard_size    INT : maximum shard size ({}) [use 0 to consider all data in a single shard]
   -max_length    INT : skip example if number of (src/tgt) tokens exceeds this ({})
   -batch_size    INT : maximum batch size ({})
   -batch_type STRING : sentences or tokens ({})

   -cuda              : use cuda device instead of cpu ({})
   -log_file     FILE : log file  (stderr)
   -log_level  STRING : log level [debug, info, warning, critical, error] (info)
   -h                 : this help
'''.format(self.prog, self.output, self.beam_size, self.n_best, self.max_size, self.alpha, self.format, self.shard_size, self.max_length, self.batch_size, self.batch_type, self.cuda))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)

  if not os.path.isdir(o.dnet):
    logging.error('unavailable network directory: {}'.format(o.dnet))
    sys.exit()
  if not os.path.isfile(o.dnet + '/network'):
    logging.error('cannot find network file: {}'.format(o.dnet + '/network'))
    sys.exit()
  if not os.path.isfile(o.dnet + '/src_pre'):
    logging.error('cannot find source preprocessor file: {}'.format(o.dnet + '/src_pre'))
    sys.exit()
  if not os.path.isfile(o.dnet + '/tgt_pre'):
    logging.error('cannot find target preprocessor file: {}'.format(o.dnet + '/tgt_pre'))
    sys.exit()

  with open(o.dnet + '/network', 'r') as f:
    n = yaml.load(f, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)
  logging.info("Network = {}".format(n))

  if n['preprocessor'] == 'sentencepiece':
    src_pre = SentencePiece(fmod=o.dnet + '/src_pre')
    tgt_pre = SentencePiece(fmod=o.dnet + '/tgt_pre')
  elif n['preprocessor'] == 'space':
    src_pre = Space(fmod=o.dnet + '/src_pre')
    tgt_pre = Space(fmod=o.dnet + '/tgt_pre')
  else:
    logging.error('Bad tokenizer optioin {}'.format(n['tokenizer']))
    sys.exit()
  assert src_pre.idx_pad == tgt_pre.idx_pad, 'src/tgt vocabularies must have the same idx_pad'

  ##################
  ### load model ###
  ##################
  device = torch.device('cuda' if o.cuda and torch.cuda.is_available() else 'cpu')
  model = Encoder_Decoder(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], len(src_pre), len(tgt_pre), src_pre.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))
  model = load_checkpoint(o.dnet + '/network', model, device)

  ##################
  ### load test ####
  ##################
  test = Dataset(src_pre, tgt_pre, o.input, None, o.shard_size, o.batch_size, o.batch_type, o.max_length)

  ##################
  ### Inference ####
  ##################
  inference = Inference(model, src_pre, tgt_pre, o, device)
  inference.translate(test,o.output)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
