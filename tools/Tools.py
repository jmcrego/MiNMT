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

  if os.path.isfile(dnet + '/joint_pre'):
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




