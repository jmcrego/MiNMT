# -*- coding: utf-8 -*-

#import io
#import os
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

##############################################################################################################
### network Params ###########################################################################################
##############################################################################################################
class network_params():

  def __init__(self):
    self.emb_dim = 4
    self.qk_dim = 2
    self.v_dim = 2
    self.ff_dim = 12
    self.n_heads = 8
    self.n_layers = 6

  def usage(self):
    return '''
  Network params
   -emb_dim : model embedding dimension
   -qk_dim : query/key dimension
   -v_dim : value dimension
   -ff_dim : feed-forward inner layer dimension
   -n_heads : number of attention heads
   -n_layers : number of encoder layers
'''

  def read_par(self, key, value):
      if key=='-network_params':
        self.read_file(value)
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
      return False

  def read_file(self, file):
    with open(file) as f: 
      for l in f:
        toks = l.rstrip().split(' ')
        if len(toks) != 2:
          logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
          sys.exit()
        else:
          if not self.read_par('-'+toks[0], toks[1]):
            logging.error('parameter {} does not exist in network params')
            sys.exit()

##############################################################################################################
### optim Params #############################################################################################
##############################################################################################################
class optim_params():

  def __init__(self):
    self.optimizer = 'adam'

  def usage(self):
    return '''
  Optim params
'''

  def read_par(self, key, value):
      if key=='-optim_params':
        self.read_file(value)
        return True
      elif key=='-optimizer':
        self.optimizer = int(value)
        return True
      return False

  def read_file(self, file):
    with open(file) as f: 
      for l in f:
        toks = l.rstrip().split(' ')
        if len(toks) != 2:
          logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
          sys.exit()
        else:
          if not self.read_par('-'+toks[0], toks[1]):
            logging.error('parameter {} does not exist in optim params')
            sys.exit()


##############################################################################################################
### data Params ##############################################################################################
##############################################################################################################
class data_params():

  def __init__(self):
    self.tokenizer = 'space'

  def usage(self):
    return '''
  Data params
'''

  def read_par(self, key, value):
      if key=='-data_params':
        self.read_file(value)
        return True
      elif key=='-tokenizer':
        self.tokenizer = int(value)
        return True
      return False

  def read_file(self, file):
    with open(file) as f: 
      for l in f:
        toks = l.rstrip().split(' ')
        if len(toks) != 2:
          logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
          sys.exit()
        else:
          if not self.read_par('-'+toks[0], toks[1]):
            logging.error('parameter {} does not exist in data params')
            sys.exit()

##############################################################################################################
### learning Params ##########################################################################################
##############################################################################################################
class learning_params():

  def __init__(self):
    self.max_updates = 5000000

  def usage(self):
    return '''
  Learning params
'''

  def read_par(self, key, value):
      if key=='-learning_params':
        self.read_file(value)
        return True
      elif key=='-max_updates':
        self.max_updates = int(value)
        return True
      return False

  def read_file(self, file):
    with open(file) as f: 
      for l in f:
        toks = l.rstrip().split(' ')
        if len(toks) != 2:
          logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
          sys.exit()
        else:
          if not self.read_par('-'+toks[0], toks[1]):
            logging.error('parameter {} does not exist in learning params')
            sys.exit()

##############################################################################################################
### inference Params #########################################################################################
##############################################################################################################
class inference_params():

  def __init__(self):
    self.beam = 5

  def usage(self):
    return '''
  Inference params
'''

  def read_par(self, key, value):
      if key=='-inference_params':
        self.read_file(value)
        return True
      if key=='-beam':
        self.beam = int(value)
        return True
      return False

  def read_file(self, file):
    with open(file) as f: 
      for l in f:
        toks = l.rstrip().split(' ')
        if len(toks) != 2:
          logging.error('bad parameter entry \'{}\' in file {}'.format(l),file)
          sys.exit()
        else:
          if not self.read_par('-'+toks[0], toks[1]):
            logging.error('parameter {} does not exist in inference params')
            sys.exit()

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

    log_file = None
    log_level = 'info'
    
    while len(sys.argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()
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
    sys.stderr.write('''usage: {} net_params opt_params data_params learning_params inference_params [-h] [-log_level LEVL] [-log_file FILE]
  -log_file    FILE : log file  (stderr)
  -log_level STRING : log level [debug, info, warning, critical, error] (info)
  -h                : this help
{}
{}
{}
{}
{}
'''.format(self.prog,self.network.usage(),self.optim.usage(),self.data.usage(),self.learning.usage(),self.inference.usage()))
    sys.exit()





    
