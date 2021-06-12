#!/usr/bin/env python3

import sys
import os
import time
import logging
import torch
#import yaml
from transformer.Dataset import Dataset, Vocab
from transformer.Model import Encoder_Decoder, load_model, numparameters
#from transformer.Model_s_s_scc_scc import Encoder_Decoder_s_s_scc_scc
#from transformer.Model_2nmt_2c import Encoder_Decoder_2nmt_2c
from transformer.Model_sxs_sc import Encoder_Decoder_sxs_sc
from transformer.Model_sxsc_sc import Encoder_Decoder_sxsc_sc
from transformer.Model_s_s_scc import Encoder_Decoder_s_s_scc
from transformer.Inference import Inference
from tools.Tools import create_logger, read_dnet

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)
    self.dnet = None
    self.input = None
    self.prefix = None
    self.xsrc = None
    self.xtgt = None
    self.output = '-'
    self.model = None
    self.beam_size = 4
    self.n_best = 1
    self.max_size = 250
    self.alpha = 0.0
    self.format = 'pt'
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
      elif tok=='-p' and len(argv):
        self.prefix = argv.pop(0)
      elif tok=='-xs' and len(argv):
        self.xsrc = argv.pop(0)
      elif tok=='-xt' and len(argv):
        self.xtgt = argv.pop(0)
      elif tok=='-o' and len(argv):
        self.output = argv.pop(0)
      elif tok=='-m' and len(argv):
        self.model = argv.pop(0)
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

      else:
        self.usage('Unrecognized {} option'.format(tok))

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
   -xs           FILE : extended source file
   -xt           FILE : extended target file
   -p            FILE : file with prefixes (force decoding)
   -o            FILE : output file ({})
   -m            FILE : use this model file (last checkpoint)

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
                          [j] input sentence ids
                          [t] target hypothesis
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
  n, src_voc, tgt_voc = read_dnet(o.dnet)
  src_voc = Vocab(src_voc)
  tgt_voc = Vocab(tgt_voc)

  ##################
  ### load model ###
  ##################
  device = torch.device('cuda' if o.cuda and torch.cuda.is_available() else 'cpu')
  if n['model_type'] == 's_sc':
    model = Encoder_Decoder(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  elif n['model_type'] == 'sxs_sc':
    model = Encoder_Decoder_sxs_sc(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], n['share_encoders'], len(source_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  elif n['model_type'] == 'sxsc_sc':
    model = Encoder_Decoder_sxsc_sc(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
  elif n['model_type'] == 's_s_scc':
    model = Encoder_Decoder_s_s_scc(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], n['share_encoders'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)

  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device))
  step, model = load_model(o.dnet + '/network', model, device, o.model)

  ##################
  ### load test ####
  ##################
  if o.prefix is not None:
    if n['model_type'] == 's_sc':
      test = Dataset([src_voc,tgt_voc], [o.input,o.prefix], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 'sxs_sc':
      test = Dataset([src_voc, tgt_voc, tgt_voc], [o.input, o.xtgt, o.prefix], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 'sxsc_sc':
      test = Dataset([src_voc, tgt_voc, tgt_voc], [o.input, o.xtgt, o.prefix], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 's_s_scc':
      test = Dataset([src_voc, tgt_voc, tgt_voc], [o.input, o.xtgt, o.prefix], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)

  else: ### without prefix
    if n['model_type'] == 's_sc':
      test = Dataset([src_voc], [o.input], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 'sxs_sc':
      test = Dataset([src_voc, tgt_voc], [o.input, o.xtgt], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 'sxsc_sc':
      test = Dataset([src_voc, tgt_voc], [o.input, o.xtgt], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)
    elif n['model_type'] == 's_s_scc':
      test = Dataset([src_voc, tgt_voc], [o.input, o.xtgt], n['accum_eos'], o.shard_size, o.batch_size, o.batch_type, o.max_length)

  ##################
  ### Inference ####
  ##################
  inference = Inference(model, src_voc, tgt_voc, o, n['accum_eos'], device)
  inference.translate(test,o.output)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
