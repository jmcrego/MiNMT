# -*- coding: utf-8 -*-

import sys
import os
import logging
#import operator
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

  for l in lines:
    idx = []
    tok = vocab.token.tokenize(l)
    tok.insert(0,vocab.str_bos)
    tok.append(vocab.str_eos)
    for t in tok:
      idx.append(vocab[t])
      ntokens += 1
      if t == vocab.idx_unk:
        nunks += 1
    toks.append(tok)
    idxs.append(idx)
    lens.append(len(idx))
  logging.info('Read {} lines ~ {} tokens ~ {} OOVs [{:.1f}%] ~ {}'.format(len(lines), ntokens, nunks, 100.0*nunks/ntokens, ftxt))
  return toks, idxs, lens

##############################################################################################################
### Batch ####################################################################################################
##############################################################################################################
class Batch():
  def __init__(self, batch_size, batch_type, idx_pad):
    super(Batch, self).__init__()
    self.batch_size = batch_size
    self.batch_type = batch_type
    self.idx_pad = idx_pad
    self.pos = []
    self.idxs_src = []
    self.idxs_tgt = []
    self.max_len_src = 0
    self.max_len_tgt = 0

  def add(self, pos, idx_src, idx_tgt):
    ### check if the new example can fit in batch (does not exceed batch_size)
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
    ### The new example fits (does not exceeds batch_size)
    self.pos.append(pos)
    self.idxs_src.append(idx_src)
    self.max_len_src = max(len(idx_src),self.max_len_src)
    self.idxs_tgt.append(idx_tgt)
    self.max_len_tgt = max(len(idx_tgt),self.max_len_tgt)
    return True

  def pad_batch(self):
    for i in range(len(self.idxs_src)):
      self.idxs_src[i] += [self.idx_pad] * (self.max_len_src - len(self.idxs_src[i]))
      self.idxs_tgt[i] += [self.idx_pad] * (self.max_len_tgt - len(self.idxs_tgt[i]))
#    print('batch')
#    for i in range(len(self.pos)):
#      print('{}\n\t{}: {}\n\t{}: {}'.format(self.pos[i], len(self.idxs_src[i]), self.idxs_src[i], len(self.idxs_tgt[i]), self.idxs_tgt[i]))
    return [self.pos, self.idxs_src, self.idxs_tgt]

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
  def __init__(self, vocab_src, ftxt_src, vocab_tgt, ftxt_tgt=None, shard_size=100000, batch_size=64, batch_type='sentences', max_length=100):    
    super(Dataset, self).__init__()
    assert vocab_src.idx_pad == vocab_tgt.idx_pad
    assert vocab_src.idx_bos == vocab_tgt.idx_bos
    assert vocab_src.idx_eos == vocab_tgt.idx_eos
    self.shard_size = shard_size
    self.batch_type = batch_type
    self.batch_size = batch_size
    self.max_length = max_length
    self.idx_pad = vocab_src.idx_pad
    _, self.idxs_src, self.lens_src = file2idx(ftxt_src, vocab_src) #no need to save txts_src
    self.pos_lens = [i for i in range(len(self.idxs_src))] #[nsents, 1]
    self.pos_lens = np.column_stack((self.pos_lens,self.lens_src)) #[nsents, 2]
    if ftxt_tgt is None: #not a bitext
      self.idxs_tgt = None
      self.lens_tgt = None
      return
    _, self.idxs_tgt, self.lens_tgt = file2idx(ftxt_tgt, vocab_tgt) #no need to save txts_tgt
    if len(self.lens_src) != len(self.lens_tgt):
      logging.error('Different number of lines in parallel dataset {}-{}'.format(len(self.lens_src),len(self.lens_tgt)))
      sys.exit()
    self.pos_lens = np.column_stack((self.pos_lens,self.lens_tgt)) #[nsents, 3]

  def build_shards_batchs(self):
    ######################
    ### randomize all data
    ######################
    np.random.shuffle(self.pos_lens)
    logging.info('Shuffled Dataset with {} examples'.format(self.pos_lens.shape[0]))
    #############################################
    ### build shards sized of shard_size examples
    #############################################
    n_examples = 0
    n_filtered = 0
    if self.shard_size == 0:
      self.shard_size = self.pos_lens.shape[0] ### all examples in one shard
    shards = []
    curr_shard = []
    for i in range(self.pos_lens.shape[0]):
      ### filter if lenght > max_length
      if self.max_length > 0 and (self.pos_lens[i][1] > self.max_length or (self.pos_lens.ndim == 3 and self.pos_lens[i][2] > self.max_length)):
        n_filtered += 1
        continue
      n_examples += 1
      curr_shard.append(self.pos_lens[i]) #[pos, lens, lent]
      if len(curr_shard) == self.shard_size: ### filled curr_shard
        shards.append(self.sort_shard(curr_shard)) ### examples in curr_shard are sorted by length
        curr_shard = []
    if len(curr_shard): ### last shard
      shards.append(self.sort_shard(curr_shard)) ### examples in curr_shard are sorted by length
      curr_shard = []
    logging.info('Built {} shards ~ {} examples ~ {} filtered [length > {}]'.format(len(shards),n_examples,n_filtered, self.max_length))
    #############################################
    ### build batchs sized of batch_size examples
    #############################################
    self.batchs = []
    for shard in shards:
      b = Batch(self.batch_size, self.batch_type, self.idx_pad) #new empty batch
      for i in range(shard.shape[0]):
        pos = shard[i] 
        idx_src = self.idxs_src[pos]
        idx_tgt = self.idxs_tgt[pos] if self.idxs_tgt is not None else []
        if not b.add(pos, idx_src, idx_tgt): #cannot continue adding in current batch b
          self.batchs.append(b.pad_batch()) #[posses, padded_src, padded_tgt]
          #new empty batch
          b = Batch(self.batch_size, self.batch_type, self.idx_pad) 
          if not b.add(pos, idx_src, idx_tgt):
            logging.error('Example {} does not fit in empty batch [Discarded]'.format(pos))
      if len(b): ### last batch
        self.batchs.append(b.pad_batch()) #[posses, padded_src, padded_tgt]
    #each batch contains up to batch_size examples with 3 items (pos, padded_idxs_src, padded_idxs_tgt)
    logging.info('Built {} batchs [{} {}]'.format(len(self.batchs), self.batch_size, self.batch_type))

  def sort_shard(self, shard):
    shard = np.asarray(shard) #[<=shard_size, 3] (3 corresponds to [pos, len_src, len_tgt])
    if self.idxs_tgt is not None:
      shard_sorted = np.lexsort((shard[:,2], shard[:,1])) # sort by shard[:,2] (len_src) then by shard[:,1] (len_tgt) (lower to higer lengths)
    else:
      shard_sorted = np.argsort(shard[:,1]) # sort by lsrc (lower to higher lenghts)
    shard = shard[shard_sorted]
    return shard[:,0] #returns only pos (lens are not used)

  def __len__(self):
    return len(self.batchs)

  def __iter__(self):
    self.build_shards_batchs()
    idx_batch = [i for i in range(len(self.batchs))]
    np.random.shuffle(idx_batch)
    logging.info('Shuffled {} batchs'.format(len(idx_batch)))
    for idx in idx_batch:
      yield self.batchs[idx]


