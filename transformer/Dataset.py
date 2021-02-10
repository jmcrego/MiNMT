# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict

def file2idx(ftxt=None, vocab=None, token=None):
  toks = []
  idxs = []
  ntokens = 0
  nunks = 0
  with open(ftxt) as f:
    lines=f.read().splitlines()

  for l in lines:
    tok = [vocab.str_bos] + token.tokenize(l) + [vocab.str_eos] 
    idx = [vocab[t] for t in tok]
    toks.append(tok)
    idxs.append(idx)
    ntokens += len(idx)
    nunks += sum([i==vocab.idx_unk for i in idx])
    print(['{}:{}'.format(tok[i],idx[i]) for i in range(len(tok))])
  logging.info('Read {} lines ~ {} tokens ~ {} OOVs [{:.2f}%] ~ {}'.format(len(lines), ntokens, nunks, 100.0*nunks/ntokens, ftxt))
  return toks, idxs


def sort_shard(shard):
  shard = np.asarray(shard) #[shard_size, 2] 
  shard_sorted = np.argsort(shard[:,1]) # sort by lsrc (lower to higher lenghts)
  shard = shard[shard_sorted]
  return shard[:,0] #returns only pos (lens are not used)

##############################################################################################################
### Batch ####################################################################################################
##############################################################################################################
class Batch():
  def __init__(self, batch_size, batch_type):
    super(Batch, self).__init__()
    self.batch_size = batch_size
    self.batch_type = batch_type
    self.idxs_pos = []
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
    self.idxs_pos.append(pos)
    self.idxs_src.append(idx_src)
    self.idxs_tgt.append(idx_tgt)
    self.max_len_src = max(len(idx_src),self.max_len_src)
    self.max_len_tgt = max(len(idx_tgt),self.max_len_tgt)
    return True

  def batch(self):
    return [self.idxs_pos, self.idxs_src, self.idxs_tgt]

  def max_lsrc(self):
    return self.max_len_src

  def max_ltgt(self):
    return self.max_len_tgt

  def __len__(self):
    return len(self.idxs_pos)

##############################################################################################################
### Dataset ##################################################################################################
##############################################################################################################
class Dataset():
  def __init__(self, vocab_src, token_src, ftxt_src, vocab_tgt, token_tgt, ftxt_tgt=None, shard_size=100000, batch_size=64, batch_type='sentences', max_length=100):    
    super(Dataset, self).__init__()
    assert vocab_src.idx_pad == vocab_tgt.idx_pad
    assert vocab_src.idx_bos == vocab_tgt.idx_bos
    assert vocab_src.idx_eos == vocab_tgt.idx_eos
    self.shard_size = shard_size
    self.batch_type = batch_type
    self.batch_size = batch_size
    self.max_length = max_length
    self.vocab_src = vocab_src
    self.vocab_tgt = vocab_tgt
    self.token_src = token_src
    self.token_tgt = token_tgt
    self.lines_src = None
    self.lines_tgt = None

    with open(ftxt_src) as f:
      self.lines_src = f.read().splitlines()
      self.idxs_src = [None] * len(self.lines_src)

    if self.shard_size == 0:
      self.shard_size = len(self.lines_src) ### all examples in one shard

    if ftxt_tgt is not None:
      with open(ftxt_tgt) as f:
        self.lines_tgt = f.read().splitlines()
        self.idxs_tgt = [None] * len(self.lines_tgt)

      if len(self.lines_src) != len(self.lines_tgt):
        logging.error('Different number of lines in parallel dataset {}-{}'.format(len(self.lines_src),len(self.lines_tgt)))
        sys.exit()

    self.idxs_pos = [i for i in range(len(self.lines_src))]

    logging.info('Read dataset with {}-{} sentences'.format(len(self.lines_src), len(self.lines_tgt)))


  def get_shard(self, firstline):
    idxs_pos = []
    lens = []
    idxs_src = []
    idxs_tgt = []
    n_filtered = 0
    n_src_tokens = 0
    n_src_unks = 0
    n_tgt_tokens = 0
    n_tgt_unks = 0
    for currline in range(firstline, len(self.idxs_pos)):
      pos = self.idxs_pos[currline]

      ### SRC ###
      if self.idxs_src[pos] is None:
        tok_src = [self.vocab_src.str_bos] + self.token_src.tokenize(self.lines_src[pos]) + [self.vocab_src.str_eos]
        idx_src = [self.vocab_src[t] for t in tok_src]
        self.idxs_src[pos] = idx_src
      else:
        idx_src = self.idxs_src[pos]

      if self.max_length and len(tok_src) > self.max_length:
        n_filtered += 1
        continue

      if self.lines_tgt is not None:
        ### TGT ###
        if self.idxs_tgt[pos] is None:
          tok_tgt = [self.vocab_tgt.str_bos] + self.token_tgt.tokenize(self.lines_tgt[pos]) + [self.vocab_tgt.str_eos] 
          idx_tgt = [self.vocab_tgt[t] for t in tok_tgt]
          self.idxs_tgt[pos] = idx_tgt
        else:
          idx_tgt = self.idxs_tgt[pos]

        if self.max_length and len(tok_tgt) > self.max_length:
          n_filtered += 1
          continue
      ###################
      ### ADD example ###
      ###################
      idxs_pos.append(pos)
      idxs_src.append(idx_src)
      lens.append(len(idx_src))
      n_src_tokens += len(idx_src)
      n_src_unks += sum([i==self.vocab_src.idx_unk for i in idx_src])
      #print(['{}:{}'.format(tok[i],idx[i]) for i in range(len(tok))])

      if self.lines_tgt is not None:
        idxs_tgt.append(idx_tgt)
        n_tgt_tokens += len(idx_tgt)
        n_tgt_unks += sum([i==self.vocab_tgt.idx_unk for i in idx_tgt])
        #print(['{}:{}'.format(tok[i],idx[i]) for i in range(len(tok))])

      if len(idxs_src) == self.shard_size:
        break

    logging.info('Built shard with {}-{} lines ~ {}-{} tokens ~ {}-{} OOVs ~ {} filtered examples'.format(len(idxs_src), len(idxs_tgt), n_src_tokens, n_tgt_tokens, n_src_unks, n_tgt_unks, n_filtered))
    return currline, lens, idxs_pos, idxs_src, idxs_tgt


  def __iter__(self):
    ##########################
    ### randomize all data ###
    ##########################
    np.random.shuffle(self.idxs_pos)
    logging.info('Shuffled Dataset with {} examples'.format(len(self.idxs_pos)))
    ####################
    ### build shards ###
    ####################
    n_shards = 0
    n_batchs = 0
    lastline = -1
    while lastline < len(self.idxs_pos):
      lastline, lens, idxs_pos, idxs_src, idxs_tgt = self.get_shard(lastline+1) 
      n_shards += 1
      ####################
      ### build batchs ###
      ####################
      shard_ordered = np.argsort(lens) # sort by lens (lower to higher lenghts)
      b = Batch(self.batch_size, self.batch_type) #empty batch
      for i in shard_ordered:
        pos = idxs_pos[i]
        idx_src = idxs_src[i]
        idx_tgt = idxs_tgt[i] if len(idxs_tgt) else []
        if not b.add(pos, idx_src, idx_tgt): ### cannot continue adding in current batch b
          if len(b):
            yield b.batch()
            n_batchs += 1
          ### start a new batch with current example [idx_src, idx_tgt]
          b = Batch(self.batch_size, self.batch_type) 
          if not b.add(pos, idx_src, idx_tgt):
            logging.warning('Example {} does not fit in empty batch [Discarded]'.format(pos))
      if len(b): ### last batch
        yield b.batch()
        n_batchs += 1

    logging.info('End dataset iteration: {} shards {} batchs'.format(n_shards,n_batchs))

