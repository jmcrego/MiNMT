import sys
import time
#import pickle
import codecs
import logging
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
    
    def __contains__(self, s):              ### implementation of the method used when invoking : entry in vocab
        if type(s) == int: 
            return s < len(self.idx_to_tok) ### testing an Idx
        return s in self.tok_to_idx         ### testing a string
    
    def __getitem__(self, s):               ### implementation of the method used when invoking : vocab[entry]
        if type(s) == int:                  ### return a string
            return self.idx_to_tok[s]
        if s in self.tok_to_idx:            ### return an index
            return self.tok_to_idx[s] 
        else:
            return self.idx_unk
        
#######################################################
### Batch #############################################
#######################################################
class Batch():
    def __init__(self, batch_size, batch_type, n_files): 
        super(Batch, self).__init__()
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.Pos = []
        self.max_lens = [0] * n_files
        assert batch_type == 'sentences' or batch_type == 'tokens', 'Bad -batch_type option'

    def getPos_and_clean(self):
        Pos = self.Pos
        self.Pos = []
        self.max_lens = [0] * len(self.max_lens)
        return Pos
        
    def add(self, pos, lens):
        ### checks wether the new example fits in current batch
        if self.batch_type == 'tokens':
            for i,l in enumerate(lens):
                if max(self.max_lens[i],l) * (len(self.Pos)+1) > self.batch_size:
                    return False
        else:
            if len(self.Pos) == self.batch_size:
                return False
        ### adds the example (pos) with lengths (lsrc, ltgt) in batch
        self.Pos.append(pos)
        for i,l in enumerate(lens):
            self.max_lens[i] = max(self.max_lens[i],l)
        return True

    def __len__(self):
        return len(self.Pos)
    
#######################################################
### Dataset ###########################################
#######################################################
class Dataset():
    def __init__(self, ftxts, vocs, shard_size=500000, batch_size=4096, batch_type='tokens', max_length=100):
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.max_length = max_length
        self.idx_bos = []
        self.idx_eos = []
        self.File_Line_Idx = [] 
        for i in range(len(ftxts)):
            voc = vocs[i]
            self.idx_bos.append(voc.idx_bos)
            self.idx_eos.append(voc.idx_eos)
            with codecs.open(ftxts[i], 'r', 'utf-8') as fd:
                Line_Idx = [[voc[t] for t in l.split()] for l in fd.read().splitlines()]
            self.File_Line_Idx.append(Line_Idx)
            logging.info('Read Corpus ({} lines) from {}'.format(len(Line_Idx),ftxts[i]))
            assert len(self.File_Line_Idx[0]) == len(self.File_Line_Idx[-1]), 'Non parallel corpus in dataset'

    def __iter__(self):
        assert len(self.File_Line_Idx) > 0, 'Empty dataset'
        n_files = len(self.File_Line_Idx)
        n_lines = len(self.File_Line_Idx[0])
        Pos = [i for i in range(n_lines)]
        np.random.shuffle(Pos)
        logging.debug('Shuffled dataset ({} examples)'.format(n_lines))
        shards = [Pos[i:i+self.shard_size] for i in range(0, n_lines, self.shard_size)]
        logging.debug('Split dataset in {} shards'.format(len(shards)))
        for s,shard in enumerate(shards):
            ###
            ### build shard
            ########################
            shard_pos = []
            shard_len = []
            for pos in shard:
                if self.max_length:
                    maxl = max([len(self.File_Line_Idx[n][pos]) for n in range(n_files)])
                    if maxl > self.max_length:
                        continue
                shard_pos.append(pos)
                shard_len.append(len(self.File_Line_Idx[0][pos]))
            logging.debug('Built shard {}/{} ({} examples)'.format(s+1,len(shards),len(shard_pos)))
            ###
            ### build batchs
            ########################
            shard_pos = np.asarray(shard_pos)
            shard_pos = shard_pos[np.argsort(shard_len)] #sort by len (lower to higher lenghts)
            logging.debug('Sorted examples by length')
            batchs = []
            b = Batch(self.batch_size, self.batch_type, n_files)
            for pos in shard_pos:
                lens = [len(self.File_Line_Idx[n][pos]) for n in range(n_files)]
                if not b.add(pos,lens):
                    if len(b):
                        batchs.append(b.getPos_and_clean()) ### save batch
                    b.add(pos,lens) ### add current example (may be discarded if it does not fit)
            if len(b):
                batchs.append(b.getPos_and_clean()) ### save batch
            logging.debug('Built {} batchs'.format(len(batchs)))
            ###
            ### yield batchs
            ########################
            idx_batchs = [i for i in range(len(batchs))]
            np.random.shuffle(idx_batchs)
            logging.debug('Shuffled {} batchs'.format(len(idx_batchs)))
            for i in idx_batchs:
                idxs_all = [] #idxs_all[0] => source batch, idxs_all[1] => target batch, ...
                for n in range(n_files):
                    idxs = []
                    for pos in batchs[i]:
                        idxs.append([self.idx_bos[n]] + self.File_Line_Idx[n][pos] + [self.idx_eos[n]])
                    idxs_all.append(idxs)
                yield batchs[i], idxs_all
            
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=10) #10:DEBUG, 20:INFO, ...
    tic = time.time()
    ftrns = sys.argv[1].split(',')
    fvocs = sys.argv[2].split(',')
    assert len(ftrns) == len(fvocs), 'Use as many corpora as vocabs comma-separated'
    data = Dataset(ftrns,fvocs)
    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc-tic))
    #pickle.dump(data, open(sys.argv[3],"wb"))


