# -*- coding: utf-8 -*-

import sys
import os
#import yaml
#import pyonmttok
import logging
import operator
import pickle
import numpy as np
from collections import defaultdict

def file2idx(ftxt=None, vocab=None):
  if vocab is None or ftxt is None:
    return None, None, None

  toks = []
  idxs = []
  lens = []
  ntokens = 0
  nunks = 0
  with open(ftxt) as f:
    lines=f.read().splitlines()
    logging.info('Read {} lines from {}'.format(len(lines), ftxt))

  for l in lines:
    idx = []
    tok = vocab.token.tokenize(l)
    for t in tok:
      idx.append(vocab[t])
      ntokens += 1
      if t == vocab.idx_unk:
        nunks += 1
    idx.insert(0,vocab.idx_bos)
    idx.append(vocab.idx_eos)
    tok.insert(0,vocab.str_bos)
    tok.append(vocab.str_eos)
    toks.append(tok)
    idxs.append(idx)
    lens.append(len(idx))
  logging.info('Found {} <unk> in {} tokens [{:.1f}%]'.format(nunks, ntokens, 100.0*nunks/ntokens))
  return toks, idxs, lens

##############################################################################################################
### Batch ####################################################################################################
##############################################################################################################
class Batch():
  def __init__(self, batch_size, batch_type):
    super(Batch, self).__init__()
    self.batch_size = batch_size
    self.batch_type = batch_type
    self.pos = []
    self.idxs_src = []
    self.idxs_tgt = []
    self.max_len_src = 0
    self.max_len_tgt = 0

  def add(self, pos, idx_src, idx_tgt):
    if self.batch_type == 'tokens':
      if max(len(idx_src),self.max_len_src) * (len(self.idxs_src)+1) > self.batch_size:
        return False
      if max(len(idx_tgt),self.max_len_tgt) * (len(self.idxs_tgt)+1) > self.batch_size:
        return False
    elif self.batch_type == 'sentences':
      if len(self.idxs_src) == self.batch_size:
        return False
    else:
      logging.error('Bad -batch_type option')
      sys.exit()

    self.pos.append(pos)
    self.idxs_src.append(idx_src)
    self.max_len_src = max(len(idx_src),self.max_len_src)
    self.idxs_tgt.append(idx_tgt)
    self.max_len_tgt = max(len(idx_tgt),self.max_len_tgt)
    return True

  def pad_batch(self, idx_pad):
    for i in range(len(self.idxs_src)):
      self.idxs_src[i] += [idx_pad] * (self.max_len_src - len(self.idxs_src[i]))
      self.idxs_tgt[i] += [idx_pad] * (self.max_len_tgt - len(self.idxs_tgt[i]))
    return self.pos, self.idxs_src, self.idxs_tgt

  def max_lsrc(self):
    return self.max_len_src

  def max_ltgt(self):
    return self.max_len_tgt

  def __len__(self):
    return len(self.idxs_src)

##############################################################################################################
### Dataset ##################################################################################################
##############################################################################################################
class Dataset():
  def __init__(self, vocab_src, vocab_tgt=None):
    super(Dataset, self).__init__()
    self.txts_src = None
    self.idxs_src = None
    self.lens_src = None
    self.txts_tgt = None
    self.idxs_tgt = None
    self.lens_tgt = None
    self.shards = None
    self.batches = None
    self.vocab_src = vocab_src
    if vocab_tgt is None:
      self.bitext = False
      return
    self.bitext = True

    self.vocab_tgt = vocab_tgt
    assert vocab_src.idx_pad == vocab_tgt.idx_pad
    assert vocab_src.idx_bos == vocab_tgt.idx_bos
    assert vocab_src.idx_eos == vocab_tgt.idx_eos

  def numberize(self, ftxt_src, ftxt_tgt=None):
    logging.info('Numberizing dataset from files {} {}'.format(ftxt_src, ftxt_tgt))
    self.txts_src, self.idxs_src, self.lens_src = file2idx(ftxt_src, self.vocab_src)
    if not self.bitext:
      return
    self.txts_tgt, self.idxs_tgt, self.lens_tgt = file2idx(ftxt_tgt, self.vocab_tgt)
    if len(self.lens_src) != len(self.lens_tgt):
      logging.error('Different number of lines in parallel data set {}-{}'.format(len(self.lens_src),len(self.lens_tgt)))
      sys.exit()

  def split_in_shards(self, shard_size=0):
    pos_lens = [i for i in range(len(self.lens_src))]
    pos_lens = np.column_stack((pos_lens,self.lens_src)) #[nsents, 2]
    if self.bitext:
      pos_lens = np.column_stack((pos_lens,self.lens_tgt)) #[nsents, 3]
    self.lens_src = None
    self.lens_tgt = None
    np.random.shuffle(pos_lens)
    if shard_size == 0:
      shard_size = len(pos_lens)
    logging.info('Shuffled Dataset {}'.format(pos_lens.shape))
    #pos_lens_lent is [n_examples, 3] or [n_examples, 2]
    #pos is the position of the example in idx_src and idx_tgt
    #len_src/len_tgt are the respective sentence lenghts
    self.shards = []
    shard = []
    for i in range(len(pos_lens)):
      shard.append(pos_lens[i]) #[pos, lens, lent]
      if len(shard) == shard_size or i==len(pos_lens)-1: ### filled shard
        shard = np.asarray(shard)
        if self.bitext:
          shard_sorted = np.lexsort((shard[:,2], shard[:,1])) # sort by len_src then len_tgt (lower to higer lengths)
        else:
          shard_sorted = np.argsort(shard[:,1]) # sort by lsrc (lower to higher lenghts)
        shard = shard[:,0] #keep only pos
        self.shards.append(shard[shard_sorted]) #sort 
        logging.info('Sorted shard #{} {}'.format(len(self.shards),self.shards[-1].shape))
        shard = []

  def split_in_batches(self, max_length=100, batch_size=64, batch_type='sentences'):
    if self.shards is None:
      self.load_shards(binfile)
    self.batches = []
    n_filtered = 0
    n_examples = 0
    for shard in self.shards:
      b = Batch(batch_size, batch_type) #new empty batch
      for i in range(len(shard)):
        pos = shard[i]
        idx_src = self.idxs_src[pos]
        if self.bitext:
          idx_tgt = self.idxs_tgt[pos]
        else:
          idx_tgt = []
        if max_length > 0 and (len(idx_src) > max_length or len(idx_tgt) > max_length):
          n_filtered += 1
          continue
        n_examples += 1
        if not b.add(pos, idx_src, idx_tgt): #cannot continue adding in current batch b
          posses, padded_src, padded_tgt = b.pad_batch(self.vocab_src.idx_pad)
          self.batches.append([posses, padded_src, padded_tgt])
          b = Batch(batch_size, batch_type) #new empty batch
          b.add(pos, idx_src, idx_tgt)
      if len(b):
        posses, padded_src, padded_tgt = b.pad_batch(self.vocab_src.idx_pad)
        self.batches.append([posses, padded_src, padded_tgt]) #last batch

    self.batches = np.asarray(self.batches)
    logging.info('Built {} batches [size={},type={}] with {} examples over {} shards, {} examples filtered by [length > {}]'.format(self.batches.shape, batch_size, batch_type, n_examples, len(self.shards), n_filtered, max_length))
    self.shards = None
    self.idxs_src = None
    self.idxs_tgt = None

  def load_shards(self, binfile):
    if binfile is None:
      logging.error('Attempt to read None binfile')
      sys.exit()
    data = pickle.load(open(binfile, 'rb'))
    self.shards, self.idxs_src, self.idxs_tgt, self.txts_src = data
    if self.idxs_tgt is None:
      self.bitext = False
    else:
      self.bitext = True
    logging.info('Loaded {} shards [{}] from binfile {}'.format(len(self.shards), 'Bitext' if self.bitext else 'Monolingual', binfile))

  def dump_shards(self, binfile):
    if binfile is None:
      logging.error('Attempt to write None binfile')
      sys.exit()
    logging.info('Dumping {} shards to binfile {}'.format(len(self.shards), binfile))
    pickle.dump([self.shards, self.idxs_src, self.idxs_tgt, self.txts_src], open(binfile, 'wb'), pickle.HIGHEST_PROTOCOL) ### self.txts_src will be used for inference

  def get_input(self, pos):
    return self.txts_src[pos]

#  def shuffle(self):
#    np.random.shuffle(self.batches)
#    logging.info('Shuffled {} batches'.format(len(self.batches)))

  def __len__(self):
    return len(self.batches)

  def __iter__(self):
    idx_batch = [i for i in range(len(self.batches))]
    np.random.shuffle(idx_batch)
    logging.info('batches shuffled')
    for idx in idx_batch:
      yield self.batches[idx]

#    for batch in self.batches:
#      yield batch


