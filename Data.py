# -*- coding: utf-8 -*-

import sys
import os
import yaml
import pyonmttok
import logging
from collections import defaultdict
#import numpy as np

####################################################################
### OpenNMTTokenizer ###############################################
####################################################################
class OpenNMTTokenizer():
  def __init__(self, fyaml=None):
    opts = {}
    if fyaml is None:
      opts['mode'] = 'space'      
    else:
      with open(fyaml) as yamlfile: 
        opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
        if 'mode' not in opts:
          logging.error('error: missing mode in tokenizer')
          sys.exit()

    mode = opts["mode"]
    del opts["mode"]
    self.tokenizer = pyonmttok.Tokenizer(mode, **opts)
    logging.info('built tokenizer mode={} {}'.format(mode,opts))

  def tokenize(self, text):
    return self.tokenizer.tokenize(text)[0]

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)

##############################################################################################################
### Vocab ####################################################################################################
##############################################################################################################
class Vocab():
  def __init__(self, file=None): #n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, src_voc_size, tgt_voc_size, pad_idx, dropout): 
    super(Vocab, self).__init__()
    self.idx_pad = 0 
    self.str_pad = '<pad>'
    self.idx_unk = 1 
    self.str_unk = '<unk>'
    self.idx_bos = 2
    self.str_bos = '<bos>'
    self.idx_eos = 3
    self.str_eos = '<eos>'
    self.idx_sep = 4
    self.str_sep = '<sep>'
    self.tok_to_idx = defaultdict()
    self.idx_to_tok = []

    if file is None:
      return
    else:
      self.read(file)

  def __len__(self):
    return len(self.idx_to_tok)

  def __iter__(self):
    for tok in self.idx_to_tok:
      yield tok

  def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
    if type(s) == int: ### testing an index
      return s>=0 and s<len(self)    
    return s in self.tok_to_idx ### testing a string

  def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
    if type(s) == int: ### input is an index, i want the string
      if s not in self:
        logging.error("key \'{}\' not found in vocab".format(s))
        sys.exit()
      return self.idx_to_tok[s] ### s exists in self.idx_to_tok
    if s not in self: ### input is a string, i want the index
      return self.idx_unk
    return self.tok_to_idx[s]

  def read(self, file):
    if not os.path.exists(file):
      logging.error('missing {} vocab file'.format(file))
      sys.exit()

    with open(file,'r') as f: 
      for l in f:
        tok = l.rstrip()
        if tok in self.tok_to_idx:
          logging.warning('Repeated vocab entry: {} [skipping]'.format(tok))
          continue
        if ' ' in tok or len(tok) == 0:
          logging.warning('Bad vocab entry: {} [skipping]'.format(tok))
          continue
        self.idx_to_tok.append(tok)
        self.tok_to_idx[tok] = len(self.tok_to_idx)

    if self.idx_to_tok[self.idx_pad] != self.str_pad:
      logging.error('vocabulary idx={} reserved for {}'.format(self.idx_pad,self.str_pad))
      sys.exit()
    if self.idx_to_tok[self.idx_unk] != self.str_unk:
      logging.error('vocabulary idx={} reserved for {}'.format(self.idx_unk,self.str_unk))
      sys.exit()
    if self.idx_to_tok[self.idx_bos] != self.str_bos:
      logging.error('vocabulary idx={} reserved for {}'.format(self.idx_bos,self.str_bos))
      sys.exit()
    if self.idx_to_tok[self.idx_eos] != self.str_eos:
      logging.error('vocabulary idx={} reserved for {}'.format(self.idx_eos,self.str_eos))
      sys.exit()
    if self.idx_to_tok[self.idx_sep] != self.str_sep:
      logging.error('vocabulary idx={} reserved for {}'.format(self.idx_sep,self.str_sep))
      sys.exit()

    logging.info('Read Vocab ({} entries) from {}'.format(len(self.idx_to_tok), file))

  def build(self, ftokconf, min_freq=1, max_size=0):
    token = OpenNMTTokenizer(ftokconf)
    ### read tokens frequency
    self.tok_to_frq = defaultdict(int)
    for l in sys.stdin:
      for tok in token.tokenize(l.strip(' \n')):
        self.tok_to_frq[tok] += 1
    ### dump vocab from tok_to_frq
    print(self.str_pad)
    print(self.str_unk)
    print(self.str_bos)
    print(self.str_eos)
    print(self.str_sep)
    n = 5
    for tok, frq in sorted(self.tok_to_frq.items(), key=lambda item: item[1], reverse=True):
      if n == max_size or frq < min_freq:
        break
      print(tok)
      n += 1
    logging.info('built vocab ({} entries)'.format(n))

