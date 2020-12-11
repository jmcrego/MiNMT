# -*- coding: utf-8 -*-

import sys
import os
import logging
from collections import defaultdict
#import yaml
#import pyonmttok
#import operator
#import pickle
#import numpy as np
#import concurrent.futures

##############################################################################################################
### Vocab ####################################################################################################
##############################################################################################################
class Vocab():
  def __init__(self, token=None, file=None): 
    super(Vocab, self).__init__()
    self.token = token
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
    #0 <pad>
    self.tok_to_idx[self.str_pad] = self.idx_pad
    self.idx_to_tok.append(self.str_pad)
    #1 <unk>
    self.tok_to_idx[self.str_unk] = self.idx_unk
    self.idx_to_tok.append(self.str_unk)
    #2 <bos>
    self.tok_to_idx[self.str_bos] = self.idx_bos
    self.idx_to_tok.append(self.str_bos)
    #3 <eos>
    self.tok_to_idx[self.str_eos] = self.idx_eos
    self.idx_to_tok.append(self.str_eos)
    #4 <sep>
    self.tok_to_idx[self.str_sep] = self.idx_sep
    self.idx_to_tok.append(self.str_sep)

    if file is not None:
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
        logging.error("Key \'{}\' not found in vocab".format(s))
        sys.exit()
      return self.idx_to_tok[s] ### s exists in self.idx_to_tok
    if s not in self: ### input is a string, i want the index
      return self.idx_unk
    return self.tok_to_idx[s]

  def read(self, file):
    if not os.path.exists(file):
      logging.error('Missing {} vocab file'.format(file))
      sys.exit()

    with open(file,'r') as f: 
      for l in f:
        tok = l.rstrip()
        if tok in self.tok_to_idx:
          continue
        if ' ' in tok or len(tok) == 0:
          logging.warning('Bad vocab entry: {} [skipping]'.format(tok))
          continue
        self.idx_to_tok.append(tok)
        self.tok_to_idx[tok] = len(self.tok_to_idx)
    logging.debug('Read Vocab ({} entries) from file {}'.format(len(self.idx_to_tok), file))

  def dump(self, file=None):
    if file is None:
      f = sys.stdout
    else:
      f = open(file,'w')
    for tok in self.idx_to_tok:
      print(tok, file=f)
    if file is not None:
      f.close()
    logging.debug('Dumped Vocab ({} entries)'.format(len(self.idx_to_tok)))

  def build(self, min_freq=1, max_size=0):
    if self.token is None:
      logging.error('No available tokenizer')
      sys.exit()
    ### read tokens frequency
    tok_to_frq = defaultdict(int)
    nlines = 0
    for l in sys.stdin:
      nlines += 1
      for tok in self.token.tokenize(l.strip(' \n')):
        tok_to_frq[tok] += 1
    logging.debug('Read {} stdin lines with {} distinct tokens'.format(nlines,len(tok_to_frq)))
    for tok, frq in sorted(tok_to_frq.items(), key=lambda item: item[1], reverse=True):
      if (max_size and len(self.idx_to_tok) == max_size) or frq < min_freq:
        break
      if tok in self.tok_to_idx: #in case reserved words appear in text
        continue
      self.tok_to_idx[tok] = len(self.tok_to_idx)
      self.idx_to_tok.append(tok)
    logging.info('Built Vocab ({} entries)'.format(len(self.idx_to_tok)))



