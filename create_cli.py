# -*- coding: utf-8 -*-

import sys
import os
import shutil
import time
import yaml
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

    self.dnet = None
    self.src_vocab = None
    self.tgt_vocab = None
    self.src_token = None
    self.tgt_token = None

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
      elif tok=="-src_vocab" and len(argv):
        self.src_vocab = argv.pop(0)
      elif tok=="-tgt_vocab" and len(argv):
        self.tgt_vocab = argv.pop(0)
      elif tok=="-src_token" and len(argv):
        self.src_token = argv.pop(0)
      elif tok=="-tgt_token" and len(argv):
        self.tgt_token = argv.pop(0)
      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    create_logger(log_file,log_level)
    if self.dnet is None:
      logging.error('missing -dnet option')
      self.usage()
    if self.src_vocab is None:
      logging.error('missing -src_vocab option')
      self.usage()
    if self.tgt_vocab is None:
      logging.error('missing -tgt_vocab option')
      self.usage()
    if self.src_token is None:
      logging.error('missing -src_token option')
      self.usage()
    if self.tgt_token is None:
      logging.error('missing -tgt_token option')
      self.usage()

  def usage(self):
    sys.stderr.write('''{} -dnet FILE [Options]
   -dnet         DIR : network ouput directory [must not exist]
   -src_vocab   FILE : source vocabulary file
   -tgt_vocab   FILE : target vocabulary file
   -src_token   FILE : source tokenizer file
   -tgt_token   FILE : target tokenizer file

   -emb_dim      INT : model embedding dimension ({})
   -qk_dim       INT : query/key embedding dimension ({})
   -v_dim        INT : value embedding dimension ({})
   -ff_dim       INT : feed-forward inner layer dimension ({})
   -n_heads      INT : number of attention heads ({})
   -n_layers     INT : number of encoder layers ({})
   -dropout    FLOAT : dropout value ({})
   -share_embeddings : share source/target embeddings ({})

   -log_file    FILE : log file  (stderr)
   -log_level STRING : log level [debug, info, warning, critical, error] (info)
   -h                : this help
'''.format(self.prog, self.emb_dim, self.qk_dim, self.v_dim, self.ff_dim, self.n_heads, self.n_layers, self.dropout, self.share_embeddings))
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
  if not os.path.isfile(opts.src_vocab):
    logging.error('cannot find source vocab file: {}'.format(opts.src_vocab))
    sys.exit()
  if not os.path.isfile(opts.tgt_vocab):
    logging.error('cannot find target vocab file: {}'.format(opts.tgt_vocab))
    sys.exit()
  if not os.path.isfile(opts.src_token):
    logging.error('cannot find source token file: {}'.format(opts.src_token))
    sys.exit()
  if not os.path.isfile(opts.tgt_token):
    logging.error('cannot find target token file: {}'.format(opts.tgt_token))
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

  shutil.copy(opts.src_vocab, opts.dnet+'/src_voc')
  logging.info('copied source vocab {} into {}/src_voc'.format(opts.src_vocab, opts.dnet))

  shutil.copy(opts.tgt_vocab, opts.dnet+'/tgt_voc')
  logging.info('copied target vocab {} into {}/tgt_voc'.format(opts.tgt_vocab, opts.dnet))

  with open(opts.src_token, 'r') as f:
    tokopts = yaml.load(f, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)
    if 'bpe_model_path' in tokopts:
      shutil.copy(tokopts['bpe_model_path'], opts.dnet+'/src_bpe')
      logging.info('copied source bpe {} into {}/src_bpe'.format(tokopts['bpe_model_path'], opts.dnet))
      tokopts['bpe_model_path'] = '{}/src_bpe'.format(opts.dnet)
      with open("{}/src_tok".format(opts.dnet), 'w') as fyaml:    
        yaml.dump(tokopts, fyaml)
      logging.info('copied source tok {} into {}/src_tok'.format(opts.src_token, opts.dnet))

  with open(opts.tgt_token, 'r') as f:
    tokopts = yaml.load(f, Loader=yaml.SafeLoader) #Loader=yaml.FullLoader)
    if 'bpe_model_path' in tokopts:
      shutil.copy(tokopts['bpe_model_path'], opts.dnet+'/tgt_bpe')
      logging.info('copied target bpe {} into {}/tgt_bpe'.format(tokopts['bpe_model_path'], opts.dnet))
      tokopts['bpe_model_path'] = '{}/tgt_bpe'.format(opts.dnet)
      with open("{}/tgt_tok".format(opts.dnet), 'w') as fyaml:
        yaml.dump(tokopts, fyaml)
      logging.info('copied target tok {} into {}/tgt_tok'.format(opts.tgt_token, opts.dnet))

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))










    
