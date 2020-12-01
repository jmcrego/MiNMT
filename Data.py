# -*- coding: utf-8 -*-

import sys
import os
import yaml
import pyonmttok
import logging
import operator
import pickle
import numpy as np
import concurrent.futures
from collections import defaultdict

def file2idx(ftxt, fvocab, ftoken):
  ldata = []
  idata = []
  vocab = Vocab(fvocab)
  idx_pad = vocab.idx_pad
  token = OpenNMTTokenizer(ftoken)
  ntokens = 0
  nunks = 0

  with open(ftxt) as f:
    lines=f.read().splitlines()
    logging.info('Read {} lines from {}'.format(len(lines), ftxt))

  for l in lines:
    toks_idx = []
    for w in token.tokenize(l):
      toks_idx.append(vocab[w])
      ntokens += 1
      if toks_idx[-1] == vocab.idx_unk:
        nunks += 1
    toks_idx.insert(0,vocab.idx_bos)
    toks_idx.append(vocab.idx_eos)
    ldata.append(toks_idx)
    idata.append(len(toks_idx))
  logging.info('Found {} <unk> in {} tokens [{:.1f}%]'.format(nunks, ntokens, 100.0*nunks/ntokens))
  return ldata, idata, vocab.idx_pad

def build_batch_idx(shard_batch, ldata_src, ldata_tgt, src_idx_pad, tgt_idx_pad):
  batch_src, batch_tgt, batch_lsrc, batch_ltgt = [], [], [], []
  max_lsrc = shard_batch[:,1].max()
  if ldata_tgt is not None:
    max_ltgt = shard_batch[:,2].max()
  for example in shard_batch:
    if ldata_tgt is not None:
      pos, lsrc, ltgt = example
    else:
      pos, lsrc = example
    batch_src.append(ldata_src[pos] + [src_idx_pad]*(max_lsrc-lsrc))
    batch_lsrc.append(lsrc)
    if ldata_tgt is not None:
      batch_tgt.append(ldata_tgt[pos] + [tgt_idx_pad]*(max_ltgt-ltgt))
      batch_ltgt.append(ltgt)
  return [batch_src, batch_tgt, batch_lsrc, batch_ltgt]

  def len2msk(l,maxl): 
    msk = torch.arange(maxl)[None, :] < l[:, None]
    return msk


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
          logging.error('Missing mode in tokenizer')
          sys.exit()

    mode = opts["mode"]
    del opts["mode"]
    self.tokenizer = pyonmttok.Tokenizer(mode, **opts)
    logging.debug('Built tokenizer mode={} {}'.format(mode,opts))

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
    logging.debug('Read Vocab ({} entries) from file {}'.format(len(self.idx_to_tok), file))


  def build(self, ftokconf, min_freq=1, max_size=0):
    token = OpenNMTTokenizer(ftokconf)
    ### read tokens frequency
    tok_to_frq = defaultdict(int)
    nlines = 0
    for l in sys.stdin:
      nlines += 1
      for tok in token.tokenize(l.strip(' \n')):
        tok_to_frq[tok] += 1
    logging.debug('Read {} stdin lines with {} distinct tokens'.format(nlines,len(tok_to_frq)))
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
  def __init__(self, fvocab_src, fvocab_tgt, ftoken_src, ftoken_tgt, ftxt_src, ftxt_tgt, shard_size, batch_size, ofile):
    super(Dataset, self).__init__()

    if ofile is not None and os.path.exists(ofile):
      self.batches = pickle.load(open(ofile+'.bin', 'rb'))
      if len(self.batches) == 0:
        logging.error('No batches found in Dataset {}'.format(ofile))
        sys.exit()
      logging.info('Read batches [{},{}] Dataset from file {}'.format(len(self.batches),len(self.batches[0][0]),ofile))
      return

    logging.info('Building Datasets from files {} {}'.format(ftxt_src, ftxt_tgt))
    ### read ldata ###
    ldata_src, len_src, src_pad = file2idx(ftxt_src, fvocab_src, ftoken_src)
    if ftxt_tgt is not None:
      ldata_tgt, len_tgt, tgt_pad = file2idx(ftxt_tgt, fvocab_tgt, ftoken_tgt)
      if len(ldata_src) != len(ldata_tgt):
        logging.error('Different number of lines in parallel data set {}-{}'.format(len(ldata_src),len(ldata_tgt)))
        sys.exit()

    pos = [i for i in range(len(ldata_src))]
    idata = np.column_stack((pos,len_src))
    if ftxt_tgt is not None:
      idata = np.column_stack((idata,len_tgt))

    ### shuffle idata
    np.random.shuffle(idata) #idata is np.ndarray
    logging.debug('Shuffled Dataset {}'.format(idata.shape))

    ### split in shards and sort each shard to minimize padding when building batches
    if shard_size == 0:
      shard_size = len(idata)

    shards = []
    for i in range(0,len(idata),shard_size):
      shard = idata[i:min(len(idata),i+shard_size)]
      pos = shard[:,0]
      lsrc = shard[:,1]
      if ftxt_tgt is not None:
        ltgt = shard[:,2]
        shard_inds_sorted = np.lexsort((lsrc, ltgt)) # sort by lsrc then ltgt 
      else:
        shard_inds_sorted = np.argsort(lsrc) # sort by lsrc

      shard = shard[shard_inds_sorted]
      shards.append(shard) #shard is ndarray
      logging.debug('Sorted shard #{} with {} examples'.format(len(shards),len(shard)))

    ### build batches
    self.batches = []
    for shard in shards:
      for i in range(0,len(shard),batch_size):
        batch = shard[i: min(len(shard),i+batch_size)]
        if ftxt_tgt is not None:
          self.batches.append(build_batch_idx(batch, ldata_src, ldata_tgt, src_pad, tgt_pad)) 
        else:
          self.batches.append(build_batch_idx(batch, ldata_src, None, src_pad, None)) 
    self.batches = np.asarray(self.batches)
    logging.info('Built {} batches'.format(len(self.batches)))

    ### shuffle batches
    np.random.shuffle(self.batches)
    logging.debug('Shuffled {} batches'.format(len(self.batches)))

    ### batches = list of batch
    ### batch = [batch_src, batch_tgt, batch_lsrc, batch_ltgt] or [batch_src, batch_lsrc]
    ### batch_src  = [src_1,  src_2,  ... src_N] 
    ### batch_tgt  = [tgt_1,  tgt_2,  ... tgt_N] 
    ### batch_lsrc = [lsrc_1, lsrc_2, ... lsrc_N] 
    ### batch_ltgt = [ltgt_1, ltgt_2, ... ltgt_N] 
    ### src_n = [ idx_1, idx_2, ..., idx_I] (already padded)
    ### tgt_n = [ idx_1, idx_2, ..., idx_J] (already padded)
    ### N is the batch size, I and J are source/target sentence lengths
    ### save batches into binary file
    if ofile is not None:
      logging.info('Saving {}'.format(ofile+'.bin'))
      pickle.dump(self.batches, open(ofile+'.bin', 'wb'), pickle.HIGHEST_PROTOCOL)


  def __len__(self):
    return len(self.batches)

  def __iter__(self):
    for batch in self.batches:
      yield batch











