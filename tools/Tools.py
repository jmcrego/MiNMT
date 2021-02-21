# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
from tools.Preprocessor import SentencePiece, Space

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


def isbinary(fin):
  try:
    with open(fin, "r") as f:
      n = 0
      for l in f:
        n += 1
        if n == 10: 
          break
      return False
  except UnicodeDecodeError: # Fond non-text data  
    return True


def read_dnet(dnet):
  if not os.path.isdir(dnet):
    logging.error('unavailable network directory: {}'.format(dnet))
    sys.exit()
  if not os.path.isfile(dnet + '/network'):
    logging.error('cannot find network file: {}'.format(dnet + '/network'))
    sys.exit()
  if not os.path.isfile(dnet + '/joint_pre') and (not os.path.isfile(dnet + '/src_pre') or not os.path.isfile(dnet + '/tgt_pre')):
    logging.error('cannot find preprocessor file/s')
    sys.exit()

  with open(dnet + '/network', 'r') as f:
    n = yaml.load(f, Loader=yaml.SafeLoader) 
  logging.info("Network = {}".format(n))

  if os.path.isfile(dnet + '/joint_pre'): ### joint pre
    if isbinary(dnet + '/joint_pre'): 
      src_pre = SentencePiece(fmod=dnet + '/joint_pre')
      tgt_pre = src_pre
    else: 
      src_pre = Space(fmod=dnet + '/joint_pre')
      tgt_pre = src_pre

  else: ### src_pre / tgt_pre
    if isbinary(dnet + '/src_pre'): 
      src_pre = SentencePiece(fmod=dnet + '/src_pre')
      tgt_pre = SentencePiece(fmod=dnet + '/tgt_pre')
    else: 
      src_pre = Space(fmod=dnet + '/src_pre')
      tgt_pre = Space(fmod=dnet + '/tgt_pre')

  assert src_pre.idx_pad == tgt_pre.idx_pad, 'src/tgt vocabularies must have the same idx_pad'
  return n, src_pre, tgt_pre


def write_dnet(o):
  if os.path.exists(o.dnet):
    logging.error('cannot create network directory: {}'.format(o.dnet))
    sys.exit()
  if not os.path.isfile(o.src_pre):
    logging.error('cannot find source preprocessor file: {}'.format(o.src_pre))
    sys.exit()
  if not os.path.isfile(o.tgt_pre):
    logging.error('cannot find target preprocessor file: {}'.format(o.tgt_pre))
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

  if o.src_pre == o.tgt_pre: ### joint pre
    shutil.copy(o.src_pre, o.dnet+'/joint_pre')
    logging.info('copied source/target preprocessor {} into {}/joint_pre'.format(o.src_pre, o.dnet))
  else: ### src_pre / tgt_pre
    shutil.copy(o.src_pre, o.dnet+'/src_pre')
    logging.info('copied source preprocessor {} into {}/src_pre'.format(o.src_pre, o.dnet))
    shutil.copy(o.tgt_pre, o.dnet+'/tgt_pre')
    logging.info('copied target preprocessor {} into {}/tgt_pre'.format(o.tgt_pre, o.dnet))


