# -*- coding: utf-8 -*-

import os
import sys
import shutil
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

def read_dnet(dnet):
  if not os.path.isdir(dnet):
    logging.error('unavailable network directory: {}'.format(dnet))
    sys.exit()
  if not os.path.isfile(dnet + '/network'):
    logging.error('cannot find network file: {}'.format(dnet + '/network'))
    sys.exit()
  if not os.path.isfile(dnet + '/src_voc'):
    logging.error('cannot find {}/src_voc file'.format(dnet))
    sys.exit()
  if not os.path.isfile(dnet + '/tgt_voc'):
    logging.error('cannot find {}/tgt_voc file'.format(dnet))
    sys.exit()

  with open(dnet + '/network', 'r') as f:
    s = f.read()
  net = eval(s)
  logging.info("Network = {}".format(net))
  src_voc = dnet + '/src_voc'
  tgt_voc = dnet + '/tgt_voc'

  return net, src_voc, tgt_voc


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

  with open(o.dnet + '/network', 'w') as f:
    f.write(str(o.net))

  shutil.copy(o.src_voc, o.dnet+'/src_voc')
  logging.info('copied source vocabulary {} into {}/src_voc'.format(o.src_voc, o.dnet))

  shutil.copy(o.tgt_voc, o.dnet+'/tgt_voc')
  logging.info('copied target vocabulary {} into {}/tgt_voc'.format(o.tgt_voc, o.dnet))

def flatten_count(ll, count):
  lflat = []
  list(map(lflat.extend, ll)) ### flattens ll into lflat
  counts = [len(lflat)] #total number of tokens
  for c in count:
    counts.append(lflat.count(c)) #frequency of token c
  return counts






