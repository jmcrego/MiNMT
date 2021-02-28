# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
#from Preprocessor import SentencePiece, Space

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


def getValue(s):
  try:
    return int(s)
  except ValueError:
    pass
  try:
    return float(s)
  except ValueError:
    pass
  if s=='True':
    return True
  if s=='False':
    return False
  return s

def read_dnet(dnet):
  if not os.path.isdir(dnet):
    logging.error('unavailable network directory: {}'.format(dnet))
    sys.exit()
  if not os.path.isfile(dnet + '/network'):
    logging.error('cannot find network file: {}'.format(dnet + '/network'))
    sys.exit()
  if not os.path.isfile(dnet + '/joint_voc') and (not os.path.isfile(dnet + '/src_voc') or not os.path.isfile(dnet + '/tgt_voc')):
    logging.error('cannot find vocabulary file/s')
    sys.exit()

  with open(dnet + '/network', 'r') as f:
    n = {}
    for l in f:
      toks = l.split()
      if len(toks) == 2:
        key = toks[0][:-1] #discard ':'
        val = getValue(toks[1])
        #print(key,val)
        n[key] = val        
      else:
        logging.error('Bad network option line: {}'.format(l))
        sys.exit()

  logging.info("Network = {}".format(n))

  if os.path.isfile(dnet + '/joint_voc'): ### joint voc
    src_voc = dnet + '/joint_voc'
    tgt_voc = src_voc
  else: ### src_voc / tgt_voc
    src_voc = dnet + '/src_voc'
    tgt_voc = dnet + '/tgt_voc'

  return n, src_voc, tgt_voc


def write_dnet(o):
  if os.path.exists(o.dnet):
    logging.error('cannot create network directory: {}'.format(o.dnet))
    sys.exit()
  if not os.path.isfile(o.src_voc):
    logging.error('cannot find source vocabulary file: {}'.format(o.src_voc))
    sys.exit()
  if not os.path.isfile(o.tgt_voc):
    logging.error('cannot find target vocabulary file: {}'.format(o.tgt_voc))
    sys.exit()

  os.mkdir(o.dnet)
  logging.info('created network directory: {}'.format(o.dnet))

  with open(o.dnet+'/network', 'w') as f:
    f.write('emb_dim: {}\n'.format(o.emb_dim))
    f.write('qk_dim: {}\n'.format(o.qk_dim))
    f.write('v_dim: {}\n'.format(o.v_dim))
    f.write('ff_dim: {}\n'.format(o.ff_dim))
    f.write('n_heads: {}\n'.format(o.n_heads))
    f.write('n_layers: {}\n'.format(o.n_layers))
    f.write('dropout: {}\n'.format(o.dropout))
    f.write('share_embeddings: {}\n'.format(o.share_embeddings))

  if o.src_voc == o.tgt_voc: ### joint voc
    shutil.copy(o.src_voc, o.dnet+'/joint_voc')
    logging.info('copied source/target vocabulary {} into {}/joint_voc'.format(o.src_voc, o.dnet))
  else: ### src_voc / tgt_voc
    shutil.copy(o.src_voc, o.dnet+'/src_voc')
    logging.info('copied source vocabulary {} into {}/src_voc'.format(o.src_voc, o.dnet))
    shutil.copy(o.tgt_voc, o.dnet+'/tgt_voc')
    logging.info('copied target vocabulary {} into {}/tgt_voc'.format(o.tgt_voc, o.dnet))


