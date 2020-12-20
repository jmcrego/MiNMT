# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

def prepare_input_src(batch_src, idx_pad, device):
  src = [torch.tensor(seq)      for seq in batch_src] 
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device)
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  return src, msk_src #, msk_tgt

##############################################################################################################
### Beam #####################################################################################################
##############################################################################################################
class BeamSearch():
  def __init__(self, model, tgt_vocab, beam_size, max_size, n_best, device):
    assert tgt_vocab.idx_pad == model.idx_pad
    self.model = model
    self.tgt_vocab = tgt_vocab
    self.beam_size = beam_size
    self.max_size = max_size
    self.n_best = n_best
    self.device = device

  def traverse(self, batch_src):
    src, msk_src = prepare_input_src(batch_src, self.tgt_vocab.idx_pad, self.device)
    K = self.beam_size #beam_size
    N = self.n_best    #nbest_size
    bs = src.shape[0]  #batch_size
    ls = src.shape[1]  #input sequence length
    Vt, ed = self.model.tgt_emb.weight.shape
    #src is [bs,ls]
    #msk_src is [bs,ls]
    z_src = self.model.encode(src, msk_src) #[bs,ls,ed]
    z_src = z_src.repeat_interleave(repeats=K, dim=0) #[bs*K,ls,ed] (repeats dimesion 0, K times)
    msk_src = msk_src.repeat_interleave(repeats=K, dim=0) #[bs*K,1,ls]

    ### initialize beam stack (it will always contain bs:batch_size and K:beam_size sentences) [initially sentences have a single word '<bos>'] with logP=0.0
    beam_hyps = torch.ones([bs*K,1], dtype=int).to(self.device) * self.tgt_vocab['<bos>'] #[bs*K,lt=1] (bs batches) (K beams) with one sentence each '<bos>' 
    beam_logP = torch.zeros([bs*K,1], dtype=torch.float32).to(self.device)                #[bs*K,lt=1]

    for lt in range(1,self.max_size+1):
      ### produced K-best hypotheses for all histories in beam_hyps (keep only hypotheses following the last token)
      y_next = self.model.decode(z_src, beam_hyps, msk_src, msk_tgt=None)[:,-1,:] #[bs*K,lt,Vt] => [bs*K,Vt]
      next_logP, next_hyps = torch.topk(y_next, k=K, dim=1) #both are [bs*K,K]
      beam_hyps, beam_logP = self.expand_beam_with_next(beam_hyps, beam_logP, next_hyps, next_logP) #both are [bs*K,lt]
      self.print(beam_hyps, beam_logP, bs, K)
      if torch.all(torch.any(beam_hyps == self.tgt_vocab.idx_eos, dim=1)): #all hypotheses have produced <eos>
        break

    sys.exit()


  def expand_beam_with_next(self, beam_hyps, beam_logP, next_hyps, next_logP):
    #beam_hyps is [bs*K,lt]
    #beam_logP is [bs*K,lt]
    #next_hyps is [bs*K,K]
    #next_logP is [bs*K,K]
    assert beam_hyps.shape[0] == beam_logP.shape[0] == next_hyps.shape[0] == next_logP.shape[0]
    assert beam_hyps.shape[1] == beam_logP.shape[1]
    assert next_hyps.shape[1] == next_logP.shape[1]
    K = next_hyps.shape[1]
    lt = beam_hyps.shape[1]
    bs = beam_hyps.shape[0] // K

    next_hyps = next_hyps.contiguous().view(bs*K*K) #[bs*K*K]
    next_logP = next_logP.contiguous().view(bs*K*K) #[bs*K*K]
    #logging.info('next_hyps = {} next_logP = {}'.format(next_hyps.shape, next_logP.shape))
    #print(next_hyps.view(bs,K*K))
    #print(next_logP.view(bs,K*K))
    #sys.exit()

    #replicate each hyp in beam K times: [bs*K,lt] => [bs*K,K*lt]
    beam_hyps = beam_hyps.repeat_interleave(repeats=K, dim=0).contiguous().view(bs*K*K,lt) #[bs*K,K*lt] => [bs*K*K,lt]
    beam_logP = beam_logP.repeat_interleave(repeats=K, dim=0).contiguous().view(bs*K*K,lt) #[bs*K,K*lt] => [bs*K*K,lt]
    logging.info('beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### extend beam hyps with new word (next)
    beam_hyps = torch.cat((beam_hyps, next_hyps.view(-1,1)), dim=-1) #[bs*K*K, lt+1]
    beam_logP = torch.cat((beam_logP, next_logP.view(-1,1)), dim=-1) #[bs*K*K, lt+1]
    logging.info('(extend) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    lt = beam_hyps.shape[1]
    beam_hyps = beam_hyps.contiguous().view(bs,K*K,lt) #[bs, K*K, lt]
    beam_logP = beam_logP.contiguous().view(bs,K*K,lt) #[bs, K*K, lt]
    logging.info('(reshape) beam_hyps = {} beam_logP = {}'.format(beam_hyps.shape, beam_logP.shape))

    ### keep the K-best of each batch (reduce K*K hyps to the K-best)
    beam_padded = self.pad_eos(beam_hyps)
    logging.info('(padding) beam_padded = {}'.format(beam_padded.shape))
    kbest_logP, kbest_hyps = torch.topk(torch.sum(beam_logP*beam_padded,dim=2), k=K, dim=1) #both are [bs, K]
    logging.info('(kbest) kbest_hyps = {} kbest_logP = {}'.format(kbest_hyps.shape, kbest_logP.shape))

    new_beam_hyps = torch.stack([beam_hyps[t][inds] for t,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(bs*K,lt)
    new_beam_logP = torch.stack([beam_logP[t][inds] for t,inds in enumerate(kbest_hyps)], dim=0).contiguous().view(bs*K,lt)
    logging.info('(stack) new_beam_hyps = {} new_beam_logP = {}'.format(new_beam_hyps.shape, new_beam_logP.shape))

#    new_beam_logP = torch.gather(beam_logP, 1, kbest_hyps).contiguous().view(bs*K)

    return new_beam_hyps, new_beam_logP


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
    #build a new column for hyps filled with <eos>
    add = torch.ones([nhyps,1], dtype=torch.long) * eos
    #print('add',add)
    #[eos]
    #[eos]
    #[eos]
    #i make sure that each row has at least one <eos>
    hyps = torch.cat((hyps,add), dim=-1)
    #print('hyps',hyps)
    #[1,eos,3,eos]
    #[1,2,eos,eos]
    #[1,2,3,eos]
    #first_eos contains the index of the first <eos> on each row in hyps
    first_eos = torch.stack( [(row==eos).nonzero().min() for row in hyps], dim=-1 )
    #print('first_eos',first_eos)
    #[1]
    #[2]
    #[3]
    #first eos has the same shape than hyps (repeat columns)
    first_eos = first_eos.repeat_interleave(repeats=hyps.shape[1], dim=0).view(hyps.shape)
    #print('first_eos',first_eos)
    #[1,1,1,1]
    #[2,2,2,2]
    #[3,3,3,3]
    x = torch.arange(hyps.shape[1]).view(1,hyps.shape[1])
    #print('x',x)
    #[0,1,2,3]
    x = x.repeat_interleave(repeats=hyps.shape[0], dim=0)
    #print('x',x)
    #[0,1,2,3]
    #[0,1,2,3]
    #[0,1,2,3]
    #pad after the first <eos> of each row and discard last column
    pad = x.le(first_eos)[:,:-1]
    #print('pad',pad)
    #[T,T,F,F]
    #[T,T,T,F]
    #[T,T,T,T]
    #back to the original size of hyps
    return pad.view(bs,N,lt)

  def print(self, beam_hyps, beam_logP, bs, K):
    #beam_hyps and beam_logP are [bs*K,lt]
    lt = beam_hyps.shape[1]
    beam_hyps = beam_hyps.view(bs,K,lt) #[bs,K,lt]
    beam_logP = beam_logP.view(bs,K,lt) #[bs,K,lt]
    pad_eos = self.pad_eos(beam_hyps)
    beam_hyps *= pad_eos
    beam_logP *= pad_eos
    #print('beam_hyps = {}'.format(beam_hyps.shape))

    for b in range(bs):
      curr_hyps = beam_hyps[b] #[K,lt]
      curr_logP = beam_logP[b] #[K,lt]
      logging.info('curr_hyps = {} curr_logP = {}'.format(curr_hyps.shape, curr_logP.shape))
      kbest_logP, kbest_hyps = torch.topk(torch.sum(curr_logP, dim=0), k=K, dim=0) #both are [bs, K]
      for h in range(len(kbest_hyps)):
        k = kbest_hyps[h]
        cost = sum(curr_logP[k])
        sys.stdout.write('step:{} batch:{} hyp:{} cost:{:.5f} |||'.format(lt,b,h,cost))
        for i in range(len(curr_hyps[k])):
          idx = curr_hyps[k,i].item()
          logP = curr_logP[k,i].item()
          wrd = self.tgt_vocab[idx]
          sys.stdout.write(' {}:{}:{:.5f}'.format(idx, wrd, logP))
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
      b = BeamSearch(self.model, self.tgt_vocab, self.beam_size, self.max_size, self.n_best, device)
      for i_batch, (batch_src, _) in enumerate(testset):
        logging.debug('Translate #batch:{}'.format(i_batch))
        b.traverse(batch_src)

 









