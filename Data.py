# -*- coding: utf-8 -*-

import sys
import os
import yaml
import pyonmttok
import logging
import random
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
  def __init__(self, file=None): 
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
          logging.warning('Repeated vocab entry: {} [skipping]'.format(tok))
          continue
        if ' ' in tok or len(tok) == 0:
          logging.warning('Bad vocab entry: {} [skipping]'.format(tok))
          continue
        self.idx_to_tok.append(tok)
        self.tok_to_idx[tok] = len(self.tok_to_idx)

    if self.idx_to_tok[self.idx_pad] != self.str_pad:
      logging.error('Vocabulary idx={} reserved for {}'.format(self.idx_pad,self.str_pad))
      sys.exit()
    if self.idx_to_tok[self.idx_unk] != self.str_unk:
      logging.error('Vocabulary idx={} reserved for {}'.format(self.idx_unk,self.str_unk))
      sys.exit()
    if self.idx_to_tok[self.idx_bos] != self.str_bos:
      logging.error('Vocabulary idx={} reserved for {}'.format(self.idx_bos,self.str_bos))
      sys.exit()
    if self.idx_to_tok[self.idx_eos] != self.str_eos:
      logging.error('Vocabulary idx={} reserved for {}'.format(self.idx_eos,self.str_eos))
      sys.exit()
    if self.idx_to_tok[self.idx_sep] != self.str_sep:
      logging.error('Vocabulary idx={} reserved for {}'.format(self.idx_sep,self.str_sep))
      sys.exit()
    logging.info('Read Vocab ({} entries) from {}'.format(len(self.idx_to_tok), file))


  def build(self, ftokconf, min_freq=1, max_size=0):
    token = OpenNMTTokenizer(ftokconf)
    ### read tokens frequency
    tok_to_frq = defaultdict(int)
    nlines = 0
    for l in sys.stdin:
      nlines += 1
      for tok in token.tokenize(l.strip(' \n')):
        tok_to_frq[tok] += 1
    logging.info('Read {} stdin lines with {} distinct tokens'.format(nlines,len(tok_to_frq)))
    ### dump vocab from tok_to_frq
    print(self.str_pad)
    print(self.str_unk)
    print(self.str_bos)
    print(self.str_eos)
    print(self.str_sep)
    n = 5
    for tok, frq in sorted(tok_to_frq.items(), key=lambda item: item[1], reverse=True):
      if n == max_size or frq < min_freq:
        break
      print(tok)
      n += 1
    logging.info('Built vocab ({} entries)'.format(n))



##############################################################################################################
### Dataset ##################################################################################################
##############################################################################################################
class Dataset():
  def __init__(self, ftok_src, ftok_tgt, vocab_src, vocab_tgt, ftxt_src, ftxt_tgt):
    super(Dataset, self).__init__()

    vdata = [] ### contains [pos, ltokens_src, ltokens_tgt]

    token = OpenNMTTokenizer(ftok_src)
    ntokens = 0
    nunks = 0
    with open(ftxt_src,'r') as f: 
      for i,l in enumerate(f):
        toks_idx = []
        for w in token.tokenize(l):
          toks_idx.append(vocab_src[w])
          ntokens += 1
          if toks_idx[-1] == vocab_src.idx_unk:
            nunks += 1
        vdata.append([i,toks_idx])
      logging.info('Read {} lines with {} tokens ({} <unk>) from {}'.format(i,ftxt_src, ntokens, nunks))

    token = OpenNMTTokenizer(ftok_tgt)
    ntokens = 0
    nunks = 0
    with open(ftxt_tgt,'r') as f: 
      for i,l in enumerate(f):
        toks_idx = []
        for w in token.tokenize(l):
          toks_idx.append(vocab_tgt[w])
          ntokens += 1
          if toks_idx[-1] == vocab_tgt.idx_unk:
            nunks += 1
        vdata[i].append(toks_idx)
      logging.info('Read {} lines with {} tokens ({} <unk>) from {}'.format(i,ftxt_tgt, ntokens, nunks))

    ### shuffle vdata
    random.shuffle(vdata)



















