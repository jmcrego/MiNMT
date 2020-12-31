# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
from collections import defaultdict
import torch

##############################################################################################################
### Greedy ###################################################################################################
##############################################################################################################
class GreedySearch():
  def __init__(self, model, tgt_vocab, max_size, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.max_size = max_size
    self.device = device

  def traverse(self, batch_src):
    Vt, ed = self.model.tgt_emb.weight.shape

    ###
    ### encode the src sequence
    ###

    src = [torch.tensor(seq) for seq in batch_src] #[bs, ls]
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=self.tgt_vocab.idx_pad).to(self.device) #src is [bs,ls]
    msk_src = (src != self.tgt_vocab.idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    bs = src.shape[0]  #batch_size

    ###
    ### decode step-by-step (produce one tgt token at each time step)
    ###

    ### initialize search with <bos> (logP=0.0)
    beam_hyps = torch.ones([bs,1], dtype=int).to(self.device) * self.tgt_vocab.idx_bos #[bs,lt=1]
    beam_logP = torch.zeros([bs,1], dtype=torch.float32).to(self.device)               #[bs,lt=1]
    beam_done = torch.zeros([bs,1], dtype=torch.bool).to(self.device) #[bs,1]

    while True:
      y_next = self.model.decode(z_src, beam_hyps, msk_src, msk_tgt=None)[:,-1,:] #[bs,lt,Vt] => [bs,Vt]
      next_logP, next_hyps = torch.topk(y_next, k=1, dim=1) #both are [bs,1]
      ### extend beam hyps with next token
      beam_hyps = torch.cat((beam_hyps, next_hyps), dim=-1) #[bs, lt+1]
      beam_logP = torch.cat((beam_logP, next_logP), dim=-1) #[bs, lt+1]
      finished = next_hyps==self.tgt_vocab.idx_eos
      beam_done = torch.logical_or(beam_done, next_hyps==self.tgt_vocab.idx_eos)
      lt = beam_hyps.shape[1]
      if lt >= self.max_size or torch.all(beam_done):
        break

    hyps = beam_hyps.numpy()
    for hyp in hyps:
      toks = [self.tgt_vocab[idx] for idx in hyp]
      print(' '.join(toks))



##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class BeamSearch():
  def __init__(self, model, tgt_vocab, beam_size, max_size, n_best, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    #self.beam_size = beam_size
    self.max_size = max_size
    self.n_best = n_best
    self.device = device

  def traverse(self, batch_src):
    K = len(batch_src) #self.beam_size #beam_size
    N = self.n_best    #nbest_size
    Vt, ed = self.model.tgt_emb.weight.shape

    self.final_hyps = [defaultdict()] * beam_size

    ###
    ### encode the src sequence
    ###

    src = [torch.tensor(seq) for seq in batch_src] #[bs, ls]
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=self.tgt_vocab.idx_pad).to(self.device) #src is [bs,ls]
    msk_src = (src != self.tgt_vocab.idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    bs = src.shape[0]  #batch_size

    ###
    ### decode step-by-step (produce one tgt token at each time step)
    ###

    ### initialize stack with <bos> (logP=0.0)
    beam_hyps = torch.ones([bs,1], dtype=int).to(self.device) * self.tgt_vocab['<bos>'] #[bs,lt=1]
    beam_logP = torch.zeros([bs,1], dtype=torch.float32).to(self.device)                #[bs,lt=1]

    ### produce first token after <bos>
    y_next = self.model.decode(z_src, beam_hyps, msk_src, msk_tgt=None)[:,-1,:]         #[bs,lt,Vt] => [bs,Vt]
    next_logP, next_hyps = torch.topk(y_next, k=K, dim=1) #both are [bs,K]
    beam_hyps, beam_logP = self.expand_beam(beam_hyps.contiguous().view(bs,1,1), beam_logP.contiguous().view(bs,1,1), next_hyps.contiguous().view(bs,1,K), next_logP.contiguous().view(bs,1,K)) #both are [bs,K,lt]
    #logging.info('beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### from now on, z_src contains K hyps per batch [bs*K,ls,ed]
    z_src = z_src.repeat_interleave(repeats=K, dim=0) #[bs*K,ls,ed] (repeats dimesion 0, K times)
    msk_src = msk_src.repeat_interleave(repeats=K, dim=0) #[bs*K,1,ls]
    #logging.info('z_src = {}'.format(z_src.shape))
    #logging.info('msk_src = {}'.format(msk_src.shape))

    for lt in range(2,self.max_size+1):
      y_next = self.model.decode(z_src, beam_hyps.contiguous().view(bs*K,lt), msk_src, msk_tgt=None)[:,-1,:] #[bs*K,lt,Vt] => [bs*K,Vt]
      next_logP, next_hyps = torch.topk(y_next, k=K, dim=1) #both are [bs*K,K]
      beam_hyps, beam_logP = self.expand_beam(beam_hyps, beam_logP, next_hyps.contiguous().view(bs,K,K), next_logP.contiguous().view(bs,K,K)) #both are [bs,K,lt]

      if torch.all(torch.any(beam_logP == -float('Inf'), dim=2)): 
        print(beam_logP)
        break
#      if torch.all(torch.any(beam_hyps == self.tgt_vocab.idx_eos, dim=2)): #all hypotheses have produced <eos>
#        break

    sys.exit()


  def expand_beam(self, beam_hyps, beam_logP, next_hyps, next_logP):
    #beam_hyps is [bs,B,lt]
    #beam_logP is [bs,B,lt]
    #next_hyps is [bs,B,K]
    #next_logP is [bs,B,K]
    assert beam_hyps.shape[0] == beam_logP.shape[0] == next_hyps.shape[0] == next_logP.shape[0] #bs
    assert beam_hyps.shape[1] == beam_logP.shape[1] == next_hyps.shape[1] == next_logP.shape[1] #B
    assert next_hyps.shape[2] == next_logP.shape[2] #K
    assert beam_hyps.shape[2] == beam_logP.shape[2] #lt
    bs = beam_hyps.shape[0]
    B = next_hyps.shape[1]
    K = next_hyps.shape[2]
    lt = beam_hyps.shape[2]
    #print(bs,B,K,lt)

    next_hyps = next_hyps.contiguous().view(bs*B*K,1) #[bs*B*K,1]
    next_logP = next_logP.contiguous().view(bs*B*K,1) #[bs*B*K,1]
    #logging.info('next_hyps = {} next_logP = {}'.format(next_hyps.shape, next_logP.shape))

    #replicate each hyp in beam K times: [bs*B,lt] => [bs*B,K*lt]
    beam_hyps = beam_hyps.contiguous().view(bs*B,lt).repeat_interleave(repeats=K, dim=0).contiguous().view(bs*B*K,lt) #[bs*B,K*lt] => [bs*B*K,lt]
    beam_logP = beam_logP.contiguous().view(bs*B,lt).repeat_interleave(repeats=K, dim=0).contiguous().view(bs*B*K,lt) #[bs*B,K*lt] => [bs*B*K,lt]
    #logging.info('beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### extend beam hyps with new word (next)
    beam_hyps = torch.cat((beam_hyps, next_hyps), dim=-1) #[bs*B*K, lt+1]
    beam_logP = torch.cat((beam_logP, next_logP), dim=-1) #[bs*B*K, lt+1]
    #logging.info('(extend) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))
    lt = beam_hyps.shape[1]

    beam_hyps = beam_hyps.contiguous().view(bs,B*K,lt) #[bs, B*K, lt]
    beam_logP = beam_logP.contiguous().view(bs,B*K,lt) #[bs, B*K, lt]
    #logging.info('(reshape) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### hyps that already produced <eos> are assigned logP=-Inf to prevent force leave the beam next step
    beam_pad = self.pad_eos(beam_hyps) #[bs, B*K, lt]
    beam_logP[beam_pad==True] = -float('Inf') #[bs, B*K, lt]
    ### save thos hyps that just produced <eos>
    self.save_hyps(beam_hyps, beam_logP)


    #keep the K-best of each batch (reduce B*K hyps to the K-best)
    kbest_logP, kbest_hyps = torch.topk(torch.sum(beam_logP,dim=2), k=K, dim=1) #both are [bs, K] (finds the K-best of dimension 1 (B*K))
    #logging.info('(kbest) kbest_hyps = {} kbest_logP = {}'.format(kbest_hyps.shape, kbest_logP.shape))

    beam_hyps = torch.stack([beam_hyps[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(bs,K,lt)
    beam_logP = torch.stack([beam_logP[b][inds] for b,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(bs,K,lt)
    #logging.info('(stack) new_beam_hyps = {} new_beam_logP = {}'.format(new_beam_hyps.shape, new_beam_logP.shape))

    return beam_hyps, beam_logP

  def save_hyps(self, beam_hyps, beam_logP):
    #beam_hyps, beam_logP, beam_pad are [bs,B*K,lt]
    bs = beam_hyps.shape[0]
    N = beam_hyps.shape[1]
    lt = beam_hyps.shape[2]

    for b in range(bs):
      for n in range(N):
        if beam_hyps[b,n,-1]==self.tgt_vocab.idx_eos:
          l = beam_hyps[b,n].tolist()
          c = sum(beam_logP[b,n]).item()
          self.final_hyps[b][' '.join(map(str,l))] = c
          print(c,[self.tgt_vocab[t] for t in l])

  def pad_eos(self, hyps):
    #hyps is [bs,N,lt]
    (bs, N, lt) = hyps.shape
    eos = self.tgt_vocab.idx_eos
    hyps = hyps.view(-1,lt) #[bs*N,lt]
    nhyps = hyps.shape[0]
    #print('hyps',hyps)
    #[1,eos,3]
    #[1,2,eos]
    #[1,2,3]
    #build a new column for hyps filled with <eos> to ensure all hyps (rows) have one <eos>
    col = torch.ones([nhyps,1], dtype=torch.long) * eos
    #print('col',col)
    #[eos]
    #[eos]
    #[eos]
    hyps = torch.cat((hyps,col), dim=-1) #[bs*N,lt+1]
    #print('hyps',hyps)
    #[1,eos,3,eos]
    #[1,2,eos,eos]
    #[1,2,3,eos]
    #first_eos contains the index of the first <eos> on each row in hyps
    first_eos = torch.stack( [(row==eos).nonzero(as_tuple=False).min() for row in hyps], dim=-1 ) #[bs*N,1]
    #print('first_eos',first_eos)
    #[1]
    #[2]
    #[3]
    #first eos has the same shape than hyps (repeat columns)
    first_eos = first_eos.repeat_interleave(repeats=hyps.shape[1], dim=0).view(hyps.shape) #[bs*N,lt+1]
    #print('first_eos',first_eos)
    #[1,1,1,1]
    #[2,2,2,2]
    #[3,3,3,3]
    x = torch.arange(hyps.shape[1]).view(1,hyps.shape[1]) #[1,lt+1]
    #print('x',x)
    #[0,1,2,3]
    x = x.repeat_interleave(repeats=hyps.shape[0], dim=0) #[bs*N,lt+1]
    #print('x',x)
    #[0,1,2,3]
    #[0,1,2,3]
    #[0,1,2,3]
    #pad after the first <eos> of each row
    pad = x.gt(first_eos) #[bs*N,lt+1]
    #print('pad',pad)
    #[F,F,T,T]
    #[F,F,F,T]
    #[F,F,F,F]
    #back to the original shape of hyps and discard last column (lt+1)
    pad = pad[:,:-1].view(bs,N,lt) #[bs,N,lt]
    return pad

  def print(self, beam_hyps, beam_logP):
    #beam_hyps and beam_logP are [bs,K,lt]
    bs = beam_hyps.shape[0]
    K = beam_hyps.shape[1]
    lt = beam_hyps.shape[2]
    beam_pad = self.pad_eos(beam_hyps)
    beam_logP[beam_pad==True] = 0.0
    print('steps = {}'.format(lt))

    for b in range(bs):
      curr_hyps = beam_hyps[b] #[K,lt]
      curr_logP = beam_logP[b] #[K,lt]
      kbest_logP, kbest_hyps = torch.topk(torch.sum(curr_logP, dim=1), k=K, dim=0) #both are [bs, K]
      curr_hyps = curr_hyps[kbest_hyps]
      curr_logP = curr_logP[kbest_hyps]
      for k in range(len(curr_hyps)):
        cost = sum(curr_logP[k])
        sys.stdout.write('step:{} batch:{} hyp:{} logP:{:.5f} |||'.format(lt,b,k,cost))
        for i in range(len(curr_hyps[k])):
          idx = curr_hyps[k,i].item()
          wrd = self.tgt_vocab[idx]
          logP = curr_logP[k,i].item()
          sys.stdout.write(' {}:{:.5f}'.format(wrd, logP))
          #sys.stdout.write(' {}:{}:{:.5f}'.format(idx, wrd, logP))
        print()

##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, tgt_vocab, oi): 
    super(Inference, self).__init__()
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = oi.beam_size
    self.max_size = oi.max_size
    self.n_best = oi.n_best

  def translate(self, testset, device):
    logging.info('Running: inference')

    with torch.no_grad():
      self.model.eval()
      #b = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.max_size, self.n_best, device)
      g = GreedySearch(self.model, self.tgt_vocab, self.max_size, device)
      for i_batch, (batch_src, _) in enumerate(testset):
        logging.debug('Translate #batch:{}'.format(i_batch))
        #b.traverse(batch_src)
        g.traverse(batch_src)

 









