# -*- coding: utf-8 -*-

import sys
import os
import time
import random
import logging
import torch
import yaml
import numpy as np
from transformer.Dataset import Dataset
from transformer.Vocab import Vocab
from tools.ONMTTokenizer import ONMTTokenizer
from transformer.Model import Encoder_Decoder, load_checkpoint_or_initialise, save_checkpoint, load_checkpoint, numparameters
from transformer.Optimizer import OptScheduler, LabelSmoothing_NLL, LabelSmoothing_KLDiv
from transformer.Learning import Learning
from tools.tools import create_logger

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)
    self.dnet = None
    self.src_train = None 
    self.tgt_train = None 
    self.src_valid = None 
    self.tgt_valid = None 
    ### learning 
    self.max_steps = 0
    self.max_epochs = 0
    self.validate_every = 5000
    self.save_every =5000
    self.report_every = 100
    self.keep_last_n = 2
    self.clip_grad_norm = 0.0
    ### optimization
    self.lr = 2.0
    self.min_lr = 0.0001
    self.beta1 = 0.9
    self.beta2 = 0.998
    self.eps = 1e-9
    self.noam_scale = 2.0
    self.noam_warmup = 4000
    self.label_smoothing = 0.1
    self.loss = 'KLDiv'
    ### data
    self.shard_size = 100000
    self.max_length = 100
    self.batch_size = 4096
    self.batch_type = 'tokens'

    self.cuda = False
    self.seed = 12345
    log_file = 'stderr'
    log_level = 'info'

    while len(argv):
      tok = argv.pop(0)
      if tok=="-h":
        self.usage()

      elif tok=='-dnet' and len(argv):
        self.dnet = argv.pop(0)
        self.dnet = self.dnet[:-1] if self.dnet[-1]=='/' else self.dnet ### remove trailing '/'
      elif tok=='-max_steps':
        self.max_steps = int(argv.pop(0))
      elif tok=='-max_epochs':
        self.max_epochs = int(argv.pop(0))
      elif tok=='-validate_every':
        self.validate_every = int(argv.pop(0))
      elif tok=='-save_every':
        self.save_every = int(argv.pop(0))
      elif tok=='-report_every':
        self.report_every = int(argv.pop(0))
      elif tok=='-keep_last_n':
        self.keep_last_n = int(argv.pop(0))
      elif tok=='-clip_grad_norm':
        self.clip_grad_norm = float(argv.pop(0))

      elif tok=='-lr':
        self.lr = float(argv.pop(0))
      elif tok=='-min_lr':
        self.min_lr = float(argv.pop(0))
      elif tok=='-beta1':
        self.beta1 = float(argv.pop(0))
      elif tok=='-beta2':
        self.beta2 = float(argv.pop(0))
      elif tok=='-eps':
        self.eps = float(argv.pop(0))
      elif tok=='-noam_scale':
        self.noam_scale = float(argv.pop(0))
      elif tok=='-noam_warmup':
        self.noam_warmup = float(argv.pop(0))
      elif tok=='-label_smoothing':
        self.label_smoothing = float(argv.pop(0))
      elif tok=='-loss':
        self.loss = argv.pop(0)

      elif tok=='-src_train':
        self.src_train = argv.pop(0)
      elif tok=='-tgt_train':
        self.tgt_train = argv.pop(0)
      elif tok=='-src_valid':
        self.src_valid = argv.pop(0)
      elif tok=='-tgt_valid':
        self.tgt_valid = argv.pop(0)
      elif tok=='-shard_size':
        self.shard_size = int(argv.pop(0))
      elif tok=='-max_length':
        self.max_length = int(argv.pop(0))
      elif tok=='-batch_size':
        self.batch_size = int(argv.pop(0))
      elif tok=='-batch_type':
        self.batch_type = argv.pop(0)

      elif tok=="-cuda":
        self.cuda = True
      elif tok=="-seed":
        self.seed = int(argv.pop(0))
      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    if self.dnet is None:
      self.usage('missing -dnet option')

    if self.src_train is None or self.tgt_train is None:
      self.usage('missing -src_train/-tgt_train options')
    create_logger(log_file,log_level)
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    logging.info("Options = {}".format(self.__dict__))

  def usage(self, messg=None):
    if messg is not None:
      sys.stderr.write(messg + '\n')
    sys.stderr.write('''usage: {} -dnet DIR -src_train FILE -tgt_train FILE -train_set FILE [-src_valid FILE -tgt_valid FILE -valid_set FILE] [Options]
   -dnet              DIR : network directory [must exist]
   -src_train        FILE : source-side training file
   -tgt_train        FILE : target-side training file
   -src_valid        FILE : source-side validation file
   -tgt_valid        FILE : target-side validation file

   [learning]
   -max_steps         INT : maximum number of training updates ({})
   -max_epochs        INT : maximum number of training epochs ({})
   -validate_every    INT : validation every INT model updates ({})
   -save_every        INT : save model every INT model updates ({})
   -report_every      INT : report every INT model updates ({})
   -keep_last_n       INT : save last INT checkpoints ({})
   -clip_grad_norm  FLOAT : clip gradients ({})

   [Optimization]
   -lr              FLOAT : initial learning rate ({})
   -min_lr          FLOAT : minimum value for learning rate ({})
   -beta1           FLOAT : beta1 for adam optimizer ({})
   -beta2           FLOAT : beta2 for adam optimizer ({})
   -eps             FLOAT : epsilon for adam optimizer ({})
   -noam_scale      FLOAT : scale of Noam decay for learning rate ({})
   -noam_warmup       INT : warmup steps of Noam decay for learning rate ({})
   -label_smoothing FLOAT : smoothing probability for label smoothing ({})
   -loss           STRING : loss function: KLDiv, NLL ({})

   [Data]
   -shard_size        INT : maximum shard size ({}) use 0 to consider all data in a single shard
   -max_length        INT : skip example if number of (src/tgt) tokens exceeds this ({})
   -batch_size        INT : maximum batch size ({})
   -batch_type     STRING : sentences or tokens ({})

   -cuda                  : use cuda device instead of cpu ({})
   -seed              INT : seed for randomness ({})
   -log_file         FILE : log file  (stderr)
   -log_level      STRING : log level [debug, info, warning, critical, error] (info)
   -h                     : this help
'''.format(self.prog, self.max_steps, self.max_epochs, self.validate_every, self.save_every, self.report_every, self.keep_last_n, self.clip_grad_norm, self.lr, self.min_lr, self.beta1, self.beta2, self.eps, self.noam_scale, self.noam_warmup, self.label_smoothing, self.loss, self.shard_size, self.max_length, self.batch_size, self.batch_type, self.cuda, self.seed))
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
  if not os.path.isfile(o.dnet + '/src_voc'):
    logging.error('cannot find source vocab file: {}'.format(o.dnet + '/src_voc'))
    sys.exit()
  if not os.path.isfile(o.dnet + '/tgt_voc'):
    logging.error('cannot find target vocab file: {}'.format(o.dnet + '/tgt_voc'))
    sys.exit()

  src_token = ONMTTokenizer(sp_model=o.dnet + '/src_tok') ### the file may not exist => space tokenizer
  src_vocab = Vocab(file=o.dnet + '/src_voc')
  tgt_token = ONMTTokenizer(sp_model=o.dnet + '/tgt_tok') ### the file may not exist => space tokenizer
  tgt_vocab = Vocab(file=o.dnet + '/tgt_voc')
  assert src_vocab.idx_pad == tgt_vocab.idx_pad, 'src/tgt vocabularies must have the same idx_pad'
  with open(o.dnet + '/network', 'r') as f:
    n = yaml.load(f, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)
  logging.info("Network = {}".format(n))

  #############################
  ### load model/optim/loss ###
  #############################
  device = torch.device('cuda' if o.cuda and torch.cuda.is_available() else 'cpu')
  model = Encoder_Decoder(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'], n['dropout'], n['share_embeddings'], len(src_vocab), len(tgt_vocab), src_vocab.idx_pad).to(device)
  logging.info('Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]), next(model.parameters()).device ))
  optim = torch.optim.Adam(model.parameters(), lr=o.lr, betas=(o.beta1, o.beta2), eps=o.eps)
  last_step, model, optim = load_checkpoint_or_initialise(o.dnet + '/network', model, optim, device)
  optScheduler = OptScheduler(optim, n['emb_dim'], o.noam_scale, o.noam_warmup, last_step)

  if o.loss == 'KLDiv':
    criter = LabelSmoothing_KLDiv(len(tgt_vocab), src_vocab.idx_pad, o.label_smoothing).to(device)
  elif o.loss == 'NLL':
    criter = LabelSmoothing_NLL(len(tgt_vocab), src_vocab.idx_pad, o.label_smoothing).to(device)
  else:
    logging.error('bad -loss option')
    sys.exit()

  ##################
  ### load data ####
  ##################
  if o.src_valid is not None:
    valid = Dataset(src_vocab, src_token, o.src_valid, tgt_vocab, tgt_token, o.tgt_valid, o.shard_size, o.batch_size, o.batch_type, o.max_length)
  else:
    valid =None
  train = Dataset(src_vocab, src_token, o.src_train, tgt_vocab, tgt_token, o.tgt_train, o.shard_size, o.batch_size, o.batch_type, o.max_length)

  n = 0
  for pos, batch_src, batch_tgt in train:
    for i in range(len(batch_src)):
      print( "{}\t{}\t{}\t{}".format(n, pos[i], batch_src[i], batch_tgt[i]) )
    n += 1
  sys.exit()

  #############
  ### learn ###
  #############
  learning = Learning(model, optScheduler, criter, o.dnet + '/network', src_vocab.idx_pad, o)
  learning.learn(train, valid, device)

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
