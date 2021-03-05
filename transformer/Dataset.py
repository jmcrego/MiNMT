# -*- coding: utf-8 -*-

import sys
import os
import logging
import codecs
import numpy as np
from collections import defaultdict

#######################################################
### Vocab #############################################
#######################################################
class Vocab():
  def __init__(self, fvoc):
    self.idx_pad = 0
    self.str_pad = '<pad>'
    self.idx_unk = 1
    self.str_unk = '<unk>'
    self.idx_bos = 2
    self.str_bos = '<bos>'
    self.idx_eos = 3
    self.str_eos = '<eos>'
    self.tok_to_idx = defaultdict()
    self.idx_to_tok = []
    with codecs.open(fvoc, 'r', 'utf-8') as fd:
      self.idx_to_tok = [l for l in fd.read().splitlines()]
    self.tok_to_idx = {k:i for i,k in enumerate(self.idx_to_tok)}
    assert self.tok_to_idx[self.str_pad] == 0, '<pad> must exist in vocab with id=0 while found id={}'.format(self.tok_to_idx[self.str_pad])
    assert self.tok_to_idx[self.str_unk] == 1, '<unk> must exist in vocab with id=1 while found id={}'.format(self.tok_to_idx[self.str_unk])
    assert self.tok_to_idx[self.str_bos] == 2, '<bos> must exist in vocab with id=2 while found id={}'.format(self.tok_to_idx[self.str_bos])
    assert self.tok_to_idx[self.str_eos] == 3, '<eos> must exist in vocab with id=3 while found id={}'.format(self.tok_to_idx[self.str_eos])
    logging.debug('Read Vocab ({} entries) from {}'.format(len(self.idx_to_tok), fvoc))

  def __len__(self):
    return len(self.idx_to_tok)

  def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
    if type(s) == int:
      return s < len(self.idx_to_tok) ### testing an Idx
    return s in self.tok_to_idx ### testing a string

  def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
    if type(s) == int: ### return a string
      return self.idx_to_tok[s]
    if s in self.tok_to_idx: ### return an index
      return self.tok_to_idx[s]
    else:
      return self.idx_unk


##############################################################################################################
### Batch ####################################################################################################
##############################################################################################################
class Batch():
  def __init__(self, batch_size, batch_type):
    super(Batch, self).__init__()
    self.batch_size = batch_size
    self.batch_type = batch_type
    self.idxs_pos = []
    self.max_lsrc = 0
    self.max_ltgt = 0

  def fits(self, lsrc, ltgt):
    ### returns True if a new example with lengths (lsrc, ltgt) can be added in this batch; False otherwise
    if self.batch_type == 'tokens':
      if max(lsrc,self.max_lsrc) * (len(self.idxs_pos)+1) > self.batch_size:
        return False
      if max(ltgt,self.max_ltgt) * (len(self.idxs_pos)+1) > self.batch_size:
        return False
    elif self.batch_type == 'sentences':
      if len(self.idxs_pos) == self.batch_size:
        return False
    else:
      logging.error('Bad -batch_type option')
      sys.exit()
    return True

  def add(self, pos, lsrc, ltgt):
    ### adds the example (pos) with lengths (lsrc, ltgt) in batch
    self.idxs_pos.append(pos)
    self.max_lsrc = max(lsrc,self.max_lsrc)
    self.max_ltgt = max(ltgt,self.max_ltgt)
    return True

  def __len__(self):
    return len(self.idxs_pos)


##############################################################################################################
### Dataset ##################################################################################################
##############################################################################################################
class Dataset():
  def __init__(self, vocs, files, shard_size=500000, batch_size=4096, batch_type='tokens', max_length=100):    
    super(Dataset, self).__init__()
    assert len(vocs) == len(files), 'Dataset must be initialized with same number of vocs/files'
    ### dataset options
    self.shard_size = shard_size
    self.batch_type = batch_type
    self.batch_size = batch_size
    self.max_length = max_length
    ### file/tokeniztion/vocabularies
    self.idx_bos = None
    self.idx_eos = None
    self.Idxs = []

    for i in range(len(files)):
      if not os.path.isfile(files[i]):
        logging.error('Cannot read file {}'.format(files[i]))
        sys.exit()
      voc = vocs[i]
      if self.idx_eos is None:
        self.idx_bos = voc.idx_bos
        self.idx_eos = voc.idx_eos

      with codecs.open(files[i], 'r', 'utf-8') as fd:
        idxs = [[voc[t] for t in l.split()] for l in fd.read().splitlines()]
      self.Idxs.append(idxs)
      ### compute tokens and OOVs
      n_tok, n_unk = flatten_count(self.Idxs, [voc.idx_unk])
      logging.info('Read Corpus ({} lines ~ {} tokens ~ {} OOVs [{:.2f}%]) from {}'.format(len(idxs),n_tok,n_unk,100.0*n_unk/n_tok,files[i]))
      assert len(self.Idxs[0]) == len(self.Idxs[-1]), 'Non-parallel corpus in dataset'

    self.idxs_src = self.Idxs[0]
    self.idxs_tgt = self.Idxs[1]
    if self.shard_size == 0:
      self.shard_size = len(self.idxs_src)
      logging.info('shard_size set to {}'.format(self.shard_size))


  def build_batchs(self, lens, idxs_pos):
    assert len(lens) == len(idxs_pos)
    ord_lens = np.argsort(lens) #sort by lens (lower to higher lenghts)
    idxs_pos = np.asarray(idxs_pos)
    idxs_pos = idxs_pos[ord_lens]

    batchs = []
    b = Batch(self.batch_size, self.batch_type) #empty batch
    for pos in idxs_pos:
      lsrc = len(self.idxs_src[pos]) + 2
      ltgt = len(self.idxs_tgt[pos]) + 2 if self.idxs_tgt is not None else 0

      if not b.fits(lsrc,ltgt): ### cannot add in current batch b
        if len(b):
          ### save batch
          batchs.append(b.idxs_pos)
          ### start a new batch 
          b = Batch(self.batch_size, self.batch_type) #empty batch

      if b.fits(lsrc,ltgt):
        ### add current example
        b.add(pos,lsrc,ltgt)
      else:
        ### discard current example
        logging.warning('Example {} does not fit in empty batch [Discarded] ~ {}:{}'.format(pos,self.ftxt_src,self.ftxt_tgt))

    if len(b): 
      ### save last batch
      batchs.append(b.idxs_pos)

    return batchs


  def get_shard(self, shard):
    ### for pos in shard:
    ### filter out examples (self.length) and returns (len, pos) of those kept
    idxs_len = []
    idxs_pos = []
    n_filtered = 0
    n_src_tokens = 0
    n_tgt_tokens = 0
    n_src_unks = 0
    n_tgt_unks = 0
    for pos in shard:
      if self.max_length and len(self.idxs_src[pos]) > self.max_length:
        n_filtered += 1
        continue

      if self.max_length and len(self.idxs_tgt[pos]) > self.max_length:
        n_filtered += 1
        continue

      ### ADD example ###
      idxs_pos.append(pos)
      idxs_len.append(len(self.idxs_src[pos]))

      n_src_tokens += len(self.idxs_src[pos])
      n_src_unks += sum([i==self.idx_unk for i in self.idxs_src[pos]])

      n_tgt_tokens += len(self.idxs_tgt[pos])
      n_tgt_unks += sum([i==self.idx_unk for i in self.idxs_tgt[pos]])

      if len(idxs_pos) == self.shard_size:
        break

    perc_src = 100.0*n_src_unks/n_src_tokens if n_src_tokens else 0.0
    perc_tgt = 100.0*n_tgt_unks/n_tgt_tokens if n_tgt_tokens else 0.0
    logging.info('Built shard with {} examples ~ {}:{} tokens ~ {}:{} OOVs [{:.2f}%:{:.2f}%] ~ {} filtered examples'.format(len(idxs_pos), n_src_tokens, n_tgt_tokens, n_src_unks, n_tgt_unks, perc_src, perc_tgt, n_filtered))
    return idxs_len, idxs_pos


  def __iter__(self):
    ### randomize all data ###
    idxs_pos = [i for i in range(len(self.idxs_src))]
    np.random.shuffle(idxs_pos)
    logging.info('Shuffled Dataset with {} examples'.format(len(idxs_pos)))
    ### split dataset in shards ###
    shards = [idxs_pos[i:i+self.shard_size] for i in range(0, len(idxs_pos), self.shard_size)]
    ### traverse shards ###
    for shard in shards: #each shard is a list of positions in the original corpus
      ### format shard ###
      lens, shard_pos = self.get_shard(shard)
      ### build batchs ###
      batchs = self.build_batchs(lens, shard_pos)
      idx_batchs = [i for i in range(len(batchs))]
      np.random.shuffle(idx_batchs)
      logging.info('Shuffled Shard with {} batchs'.format(len(idx_batchs)))
      for i in idx_batchs:
        batch_pos = batchs[i]
        idxs_src = []
        idxs_tgt = []
        for pos in batch_pos:
          idxs_src.append([self.idx_bos] + self.idxs_src[pos] + [self.idx_eos]) 
          idxs_tgt.append([self.idx_bos] + self.idxs_tgt[pos] + [self.idx_eos])
        yield batch_pos, [idxs_src, idxs_tgt]






