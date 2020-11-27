# -*- coding: utf-8 -*-

import sys
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

def read_file_param(file, pars):
  with open(file) as f: 
    for l in f:
      toks = l.rstrip().split(' ')
      if len(toks) != 2:
        logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
        sys.exit()
      else:
        if not pars.read_par('-'+toks[0], toks[1]):
          logging.error('parameter {} does not allowed in {}'.format(key, pars.__class__.__name__))
          sys.exit()


##############################################################################################################
### network Params ###########################################################################################
##############################################################################################################
class network_params():

  def __init__(self):
    self.emb_dim = 512
    self.qk_dim = 64
    self.v_dim = 64
    self.ff_dim = 1024
    self.n_heads = 8
    self.n_layers = 6

  def usage(self):
    return '''
  Network params
   -emb_dim    INT : model embedding dimension ({})
   -qk_dim     INT : query/key dimension ({})
   -v_dim      INT : value dimension ({})
   -ff_dim     INT : feed-forward inner layer dimension ({})
   -n_heads    INT : number of attention heads ({})
   -n_layers FLOAT : number of encoder layers ({})'''.format(self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers)

  def read_par(self, key, value):
      if key=='-network_params':
        read_file_param(value, self)
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
        self.n_layers = float(value)
        return True
      return False

##############################################################################################################
### optim Params #############################################################################################
##############################################################################################################
class optim_params():

  def __init__(self):
    self.optimizer = 'adam'
    self.dropout = 0.3
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
  Optim params
   -optimizer      STRING : optimization algorithm ({})
   -dropout         FLOAT : dropout value ({})
   -lr              FLOAT : initial learning rate ({})
   -min_lr          FLOAT : minimum value for learning rate ({})
   -beta1           FLOAT : beta1 for adam optimizer ({})
   -beta2           FLOAT : beta2 for adam optimizer ({})
   -eps             FLOAT : epsilon for adam optimizer ({})
   -noam_scale      FLOAT : scale of Noam decay for learning rate ({})
   -noam_warmup       INT : warmup steps of Noam decay for learning rate ({})
   -label_smoothing FLOAT : smoothing probability for label smoothing ({})'''.format(self.optimizer, self.dropout, self.lr, self.min_lr, self.beta1, self.beta2, self.eps, self.noam_scale, self.noam_warmup, self.label_smoothing)

  def read_par(self, key, value):
      if key=='-optim_params':
        read_file_param(value, self)
        return True
      elif key=='-optimizer':
        self.optimizer = int(value)
        return True
      elif key=='-dropout':
        self.dropout = float(value)
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
### data Params ##############################################################################################
##############################################################################################################
class data_params():

  def __init__(self):
    self.tokenizer = 'space'
    self.src_voc_size = 30000
    self.tgt_voc_size = 30000
    self.pad_idx = 0


  def usage(self):
    return '''
  Data params
   -tokenizer STRING : tokenizer (space)'''.format(self.tokenizer)

  def read_par(self, key, value):
      if key=='-data_params':
        self.read_file_param(value, self)
        return True
      elif key=='-tokenizer':
        self.tokenizer = value
        return True
      return False

##############################################################################################################
### learning Params ##########################################################################################
##############################################################################################################
class learning_params():

  def __init__(self):
    self.max_updates = 5000000
    self.batch_size = 32
    self.batch_type = 'sentences'    

  def usage(self):
    return '''
  Learning params
   -max_updates   INT : maximum number of training updates ({})
   -batch_size    INT : maximum batch size ({})
   -batch_type STRING : sentences or tokens ({})'''.format(self.max_updates, self.batch_size, self.batch_type)

  def read_par(self, key, value):
      if key=='-learning_params':
        self.read_file_param(value, self)
        return True
      elif key=='-max_updates':
        self.max_updates = int(value)
        return True
      elif key=='-batch_size':
        self.batch_size = int(value)
        return True
      elif key=='-batch_type':
        self.batch_type = value
        return True

      return False

##############################################################################################################
### inference Params #########################################################################################
##############################################################################################################
class inference_params():

  def __init__(self):
    self.beam = 5

  def usage(self):
    return '''
  Inference params
   -beam INT : size of beam ({})'''.format(self.beam)

  def read_par(self, key, value):
      if key=='-inference_params':
        self.read_file_param(value, self)
        return True
      if key=='-beam':
        self.beam = int(value)
        return True

      return False

##############################################################################################################
### Params ###################################################################################################
##############################################################################################################
class Params():

  def __init__(self, argv):
    self.network = network_params()
    self.optim = optim_params()
    self.data = data_params()
    self.learning = learning_params()
    self.inference = inference_params()
    self.prog = argv.pop(0)
    self.run = None

    log_file = None
    log_level = 'info'
    
    while len(sys.argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()
      elif tok=="-run" and len(sys.argv):
        self.run = sys.argv.pop(0).lower()
      elif tok=="-log_file" and len(sys.argv):
        log_file = sys.argv.pop(0)
      elif tok=="-log_level" and len(sys.argv):
        log_level = sys.argv.pop(0)

      else:
        if len(sys.argv):
          value = sys.argv.pop(0)
          if self.network.read_par(tok, value):
            continue
          elif self.optim.read_par(tok, value):
            continue
          elif self.data.read_par(tok, value):
            continue
          elif self.learning.read_par(tok, value):
            continue
          elif self.inference.read_par(tok, value):
            continue
        #error
        sys.stderr.write('error: unparsed {} option\n'.format(tok))
        self.usage()

    create_logger(log_file,log_level)
    logging.info("Network Params = {}".format(self.network.__dict__))
    logging.info("Optim Params = {}".format(self.optim.__dict__))
    logging.info("Data Params = {}".format(self.data.__dict__))
    logging.info("Learning Params = {}".format(self.learning.__dict__))
    logging.info("Inference Params = {}".format(self.inference.__dict__))

  def usage(self):
    sys.stderr.write('''usage: {} -run COMMAND [net_params] [opt_params] [data_params] [learning_params] [inference_params] [-h] [-log_level LEVL] [-log_file FILE]
   -run COMMAND : learn or inference
   -log_file FILE : log file  (stderr)
   -log_level STRING : log level [debug, info, warning, critical, error] (info)
   -h                : this help
{}
{}
{}
{}
{}
'''.format(self.prog,self.network.usage(),self.optim.usage(),self.data.usage(),self.learning.usage(),self.inference.usage()))
    sys.exit()





    
