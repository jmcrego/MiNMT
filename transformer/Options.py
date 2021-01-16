# -*- coding: utf-8 -*-

import sys
import os
import yaml
import random
import logging

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

def read_file_options(file, opts):
  with open(file, 'r') as fyaml:      
    dopts = yaml.load(fyaml, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)

  for key, value in dopts.iteritems():
    if not opts.read_opt('-'+key, value):
      logging.error('option {} does not allowed in {}'.format(key, opts.__class__.__name__))
      sys.exit()

def read_file_options2(file, opts):
  with open(file) as f: 
    for l in f:
      toks = l.rstrip().split(' ')
      if len(toks) != 2:
        logging.error('bad option entry \'{}\' in file {}'.format(l),file)
        sys.exit()
      else:
        if not opts.read_opt('-'+toks[0], toks[1]):
          logging.error('option {} does not allowed in {}'.format(key, opts.__class__.__name__))
          sys.exit()


##############################################################################################################
### network Options ##########################################################################################
##############################################################################################################
class network_options():

  def __init__(self):
    self.emb_dim = 512
    self.qk_dim = 64
    self.v_dim = 64
    self.ff_dim = 2048
    self.n_heads = 8
    self.n_layers = 6
    self.dropout = 0.1

  def usage(self):
    return '''
  Network options
   -network_options YAML : yaml file with network options
   -emb_dim          INT : model embedding dimension ({})
   -qk_dim           INT : query/key dimension ({})
   -v_dim            INT : value dimension ({})
   -ff_dim           INT : feed-forward inner layer dimension ({})
   -n_heads          INT : number of attention heads ({})
   -n_layers       FLOAT : number of encoder layers ({})
   -dropout        FLOAT : dropout value ({})'''.format(self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers, self.dropout)

  def read_opt(self, key, value):
      if key=='-network_options':
        read_file_options(value, self)
        return True
      elif key=='-emb_dim':
        self.emb_dim = int(value)
        return True
      elif key=='-qk_dim':
        self.qk_dim = int(value)
        return True
      elif key=='-v_dim':
        self.v_dim = int(value)
        return True
      elif key=='-ff_dim':
        self.ff_dim = int(value)
        return True
      elif key=='-n_heads':
        self.n_heads = int(value)
        return True
      elif key=='-n_layers':
        self.n_layers = int(value)
        return True
      elif key=='-dropout':
        self.dropout = float(value)
        return True
      return False

##############################################################################################################
### optim Options ############################################################################################
##############################################################################################################
class optim_options():

  def __init__(self):
    self.lr = 2.0
    self.min_lr = 0.0001
    self.beta1 = 0.9
    self.beta2 = 0.998
    self.eps = 1e-9
    self.noam_scale = 2.0
    self.noam_warmup = 4000
    self.label_smoothing = 0.1

  def usage(self):
    return '''
  Optim options
   -optim_options    YAML : yaml file with optim options
   -lr              FLOAT : initial learning rate ({})
   -min_lr          FLOAT : minimum value for learning rate ({})
   -beta1           FLOAT : beta1 for adam optimizer ({})
   -beta2           FLOAT : beta2 for adam optimizer ({})
   -eps             FLOAT : epsilon for adam optimizer ({})
   -noam_scale      FLOAT : scale of Noam decay for learning rate ({})
   -noam_warmup       INT : warmup steps of Noam decay for learning rate ({})
   -label_smoothing FLOAT : smoothing probability for label smoothing ({})'''.format(self.lr, self.min_lr, self.beta1, self.beta2, self.eps, self.noam_scale, self.noam_warmup, self.label_smoothing)

  def read_opt(self, key, value):
      if key=='-optim_options':
        read_file_options(value, self)
        return True
      elif key=='-lr':
        self.lr = float(value)
        return True
      elif key=='-min_lr':
        self.min_lr = float(value)
        return True
      elif key=='-beta1':
        self.beta1 = float(value)
        return True
      elif key=='-beta2':
        self.beta2 = float(value)
        return True
      elif key=='-eps':
        self.eps = float(value)
        return True
      elif key=='-noam_scale':
        self.noam_scale = float(value)
        return True
      elif key=='-noam_warmup':
        self.noam_warmup = float(value)
        return True
      elif key=='-label_smoothing':
        self.label_smoothing = float(value)
        return True

      return False

##############################################################################################################
### data Options #############################################################################################
##############################################################################################################
class data_options():

  def __init__(self):
    self.src_token = None
    self.tgt_token = None
    self.src_vocab = None 
    self.tgt_vocab = None 
    self.src_train = None 
    self.tgt_train = None 
    self.src_valid = None 
    self.tgt_valid = None 
    self.src_test = None 
    self.tgt_test = None 
    self.train_set = None
    self.valid_set = None
    self.test_set = None
    self.shard_size = 100000
    self.max_length = 0
    self.batch_size = 4096
    self.batch_type = 'tokens'    

  def usage(self):
    return '''
  Data options
   -data_options YAML : yaml file with data options
   -src_token    FILE : source-side onmt tokenizer config file ('space' mode)
   -tgt_token    FILE : target-side onmt tokenizer config file ('space' mode)
   -src_vocab    FILE : source-side vocabulary file
   -tgt_vocab    FILE : target-side vocabulary file
   -src_train    FILE : source-side training file
   -tgt_train    FILE : target-side training file
   -src_valid    FILE : source-side validation file
   -tgt_valid    FILE : target-side validation file
   -src_test     FILE : source-side test file
   -tgt_test     FILE : target-side test file
   -train_set    FILE : training dataset is read/written from/into FILE.bin
   -valid_set    FILE : validation dataset is read/written from/into FILE.bin
   -test_set     FILE : test dataset is read/written from/into FILE.bin
   -shard_size    INT : maximum shard size ({}) use 0 to consider all data in a single shard
   -max_length    INT : max number of tokens for src/tgt sentences ({})
   -batch_size    INT : maximum batch size ({})
   -batch_type STRING : sentences or tokens ({})'''.format(self.shard_size, self.max_length, self.batch_size, self.batch_type)

  def read_opt(self, key, value):
      if key=='-data_options':
        read_file_options(value, self)
        return True
      elif key=='-src_token':
        self.src_token = value
        return True
      elif key=='-tgt_token':
        self.tgt_token = value
        return True
      elif key=='-src_vocab':
        self.src_vocab = value
        return True
      elif key=='-tgt_vocab':
        self.tgt_vocab = value
        return True
      elif key=='-src_train':
        self.src_train = value
        return True
      elif key=='-tgt_train':
        self.tgt_train = value
        return True
      elif key=='-src_valid':
        self.src_valid = value
        return True
      elif key=='-tgt_valid':
        self.tgt_valid = value
        return True
      elif key=='-src_test':
        self.src_test = value
        return True
      elif key=='-tgt_test':
        self.tgt_test = value
        return True
      elif key=='-train_set':
        self.train_set = value
        return True
      elif key=='-valid_set':
        self.valid_set = value
        return True
      elif key=='-test_set':
        self.test_set = value
        return True
      elif key=='-shard_size':
        self.shard_size = int(value)
        return True
      elif key=='-max_length':
        self.max_length = int(value)
        return True
      elif key=='-batch_size':
        self.batch_size = int(value)
        return True
      elif key=='-batch_type':
        self.batch_type = value
        return True

      return False

##############################################################################################################
### learning Options #########################################################################################
##############################################################################################################
class learning_options():

  def __init__(self):
    self.max_steps = 0
    self.max_epochs = 0
    self.validate_every = 5000
    self.save_every =5000
    self.report_every = 100
    self.keep_last_n = 10
    self.clip_grad_norm = 0.0

  def usage(self):
    return '''
  Learning options
   -learning_options YAML : yaml file with learning options
   -max_steps         INT : maximum number of training updates ({})
   -max_epochs        INT : maximum number of training epochs ({})
   -validate_every    INT : validation every INT model updates ({})
   -save_every        INT : save model every INT model updates ({})
   -report_every      INT : report every INT model updates ({})
   -keep_last_n       INT : save last INT checkpoints ({})
   -clip_grad_norm  FLOAT : clip gradients ({})'''.format(self.max_steps, self.max_epochs, self.validate_every, self.save_every, self.report_every, self.keep_last_n, self.clip_grad_norm)

  def read_opt(self, key, value):
      if key=='-learning_options':
        read_file_options(value, self)
        return True
      elif key=='-max_steps':
        self.max_steps = int(value)
        return True
      elif key=='-max_epochs':
        self.max_epochs = int(value)
        return True
      elif key=='-validate_every':
        self.validate_every = int(value)
        return True
      elif key=='-save_every':
        self.save_every = int(value)
        return True
      elif key=='-report_every':
        self.report_every = int(value)
        return True
      elif key=='-keep_last_n':
        self.keep_last_n = int(value)
        return True
      elif key=='-clip_grad_norm':
        self.clip_grad_norm = float(value)
        return True

      return False

##############################################################################################################
### inference Options ########################################################################################
##############################################################################################################
class inference_options():

  def __init__(self):
    self.beam_size = 5
    self.n_best = 1
    self.max_size = 250
    self.format = 'iH'

  def usage(self):
    return '''
  Inference options
   -inference_options YAML : yaml file with inference options
   -beam_size          INT : size of beam ({})
   -n_best             INT : return n-best translation hypotheses ({})
   -max_size           INT : max hypothesis size ({})
   -format          STRING : format of output lines (default {})
                              [i] index in test set
                              [n] rank in n-best
                              [c] global hypothesis cost
                              [s] source sentence
                              [S] source sentence (detokenised)
                              [h] hypothesis
                              [H] hypothesis (detokenised)'''.format(self.beam_size, self.n_best, self.max_size, self.format)

  def read_opt(self, key, value):
      if key=='-inference_options':
        read_file_options(value, self)
        return True
      if key=='-beam_size':
        self.beam_size = int(value)
        return True
      if key=='-n_best':
        self.n_best = int(value)
        return True
      if key=='-max_size':
        self.max_size = int(value)
        return True
      if key=='-format':
        self.format = value
        return True

      return False

##############################################################################################################
### Options ##################################################################################################
##############################################################################################################
class Options():

  def __init__(self, argv):
    self.network = network_options()
    self.optim = optim_options()
    self.data = data_options()
    self.learning = learning_options()
    self.inference = inference_options()
    self.prog = argv.pop(0)

    self.suffix = None
    self.cuda = False
    log_file = None
    log_level = 'info'
    seed = 12345

    while len(sys.argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()
      elif tok=="-suffix" and len(sys.argv):
        self.suffix = sys.argv.pop(0)
      elif tok=="-cuda":
        self.cuda = True
      elif tok=="-log_file" and len(sys.argv):
        log_file = sys.argv.pop(0)
      elif tok=="-log_level" and len(sys.argv):
        log_level = sys.argv.pop(0)
      elif tok=="-seed" and len(sys.argv):
        seed = int(sys.argv.pop(0))

      else:
        if len(sys.argv):
          value = sys.argv.pop(0)
          if self.network.read_opt(tok, value):
            continue
          elif self.optim.read_opt(tok, value):
            continue
          elif self.data.read_opt(tok, value):
            continue
          elif self.learning.read_opt(tok, value):
            continue
          elif self.inference.read_opt(tok, value):
            continue
        #error
        sys.stderr.write('error: unparsed {} option\n'.format(tok))
        self.usage()

    create_logger(log_file,log_level)

    if self.suffix is None:
      logging.error("Missing -suffix option")
      self.usage()

    if os.path.exists(self.suffix+'.network'):
      logging.info('Replacing network options found in {}.network'.format(self.suffix))
      with open("{}.network".format(self.suffix), 'r') as fyaml:      
        self.network.__dict__ = yaml.load(fyaml, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)
    else:
      with open("{}.network".format(self.suffix), 'w') as fyaml:      
        yaml.dump(self.network.__dict__, fyaml) #, sort_keys=False) #default_flow_style=False)

    logging.info("Network options = {}".format(self.network.__dict__))
    logging.info("Optim options = {}".format(self.optim.__dict__))
    logging.info("Data options = {}".format(self.data.__dict__))
    logging.info("Learning options = {}".format(self.learning.__dict__))
    logging.info("Inference options = {}".format(self.inference.__dict__))
    logging.info("Suffix = {}".format(self.suffix))
    logging.info("seed = {}".format(seed))
    random.seed(seed)

  def usage(self):
    sys.stderr.write('''usage: {} -suffix FILE [net_options] [opt_options] [data_options] [learning_options] [inference_options] [-h] [-log_level LEVL] [-log_file FILE]
   -suffix    STRING : suffix for model/optim files
   -cuda             : use cuda device instead of cpu
   -log_file    FILE : log file  (stderr)
   -log_level STRING : log level [debug, info, warning, critical, error] (info)
   -seeed        INT : seed for randomness (12345)
   -h                : this help
{}
{}
{}
{}
{}
'''.format(self.prog,self.network.usage(),self.optim.usage(),self.data.usage(),self.learning.usage(),self.inference.usage()))
    sys.exit()





    
