# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict

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
  def __init__(self, spm_src, spm_tgt, ftxt_src, ftxt_tgt=None, shard_size=100000, batch_size=64, batch_type='sentences', max_length=100):    
    super(Dataset, self).__init__()
    assert spm_src.idx_pad == spm_tgt.idx_pad
    assert spm_src.idx_bos == spm_tgt.idx_bos
    assert spm_src.idx_eos == spm_tgt.idx_eos
    ### dataset options
    self.shard_size = shard_size
    self.batch_type = batch_type
    self.batch_size = batch_size
    self.max_length = max_length
    ### file/tokeniztion/vocabularies
    self.ftxt_src = ftxt_src
    self.ftxt_tgt = ftxt_tgt
    self.spm_src = spm_src
    self.spm_tgt = spm_tgt
    ### original corpora
    self.txts_src = None #list of strings (original sentences)
    self.txts_tgt = None #list of strings (original sentences)
    self.idxs_src = None #list of list of ints
    self.idxs_tgt = None #list of list of ints

    if ftxt_src is not None:
      logging.info('Reading {}'.format(ftxt_src))
      self.txts_src, self.idxs_src = self.spm_src.encode(ftxt_src,int)

    if ftxt_tgt is not None:
      logging.info('Reading {}'.format(ftxt_tgt))
      self.txts_tgt, self.idxs_tgt = self.spm_tgt.encode(ftxt_tgt,int)
      assert len(self.txts_src) == len(self.txts_tgt), 'Different number of lines in parallel dataset {}:{}'.format(len(self.txts_src),len(self.txts_tgt))

    if self.shard_size == 0:
      self.shard_size = len(self.txts_src)
      logging.info('shard_size set to {}'.format(self.shard_size))

    logging.info('Read Dataset with {} sentences {}:{}'.format(len(self.txts_src), ftxt_src, ftxt_tgt))


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
        logging.warning('Example {} does not fit in empty batch [Discarded] {}:{}'.format(pos,self.ftxt_src,self.ftxt_tgt))

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

      if self.txts_tgt is not None:
        if self.max_length and len(self.idxs_tgt[pos]) > self.max_length:
          n_filtered += 1
          continue

      ### ADD example ###
      idxs_pos.append(pos)
      idxs_len.append(len(self.idxs_src[pos]))

      n_src_tokens += len(self.idxs_src[pos])
      n_src_unks += sum([i==self.spm_src.idx_unk for i in self.idxs_src[pos]])
      #print(['{}:{}'.format(tok[i],idx[i]) for i in range(len(tok))])

      if self.txts_tgt is not None:
        n_tgt_tokens += len(self.idxs_tgt[pos])
        n_tgt_unks += sum([i==self.spm_tgt.idx_unk for i in self.idxs_tgt[pos]])
        #print(['{}:{}'.format(tok[i],idx[i]) for i in range(len(tok))])

      if len(idxs_pos) == self.shard_size:
        break

    logging.info('Built shard with {} examples ~ {}:{} tokens ~ {}:{} OOVs [{:.2f}%:{:.2f}%] ~ {} filtered examples {}:{}'.format(len(idxs_pos), n_src_tokens, n_tgt_tokens, n_src_unks, n_tgt_unks, 100.0*n_src_unks/n_src_tokens, 100.0*n_tgt_unks/n_tgt_tokens, n_filtered, self.ftxt_src, self.ftxt_tgt))
    return idxs_len, idxs_pos


  def __iter__(self):
    ### randomize all data ###
    idxs_pos = [i for i in range(len(self.txts_src))]
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
      logging.info('Shuffled shard with {} batchs'.format(len(idx_batchs)))
      for i in idx_batchs:
        batch_pos = batchs[i]
        idxs_src = []
        idxs_tgt = []
        for pos in batch_pos:
          idxs_src.append([self.spm_src.idx_bos] + self.idxs_src[pos] + [self.spm_src.idx_eos]) 
          if self.idxs_tgt is not None:
            idxs_tgt.append([self.spm_tgt.idx_bos] + self.idxs_tgt[pos] + [self.spm_tgt.idx_eos])
        yield [batch_pos, idxs_src, idxs_tgt]






