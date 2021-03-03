# -*- coding: utf-8 -*-

import sys
import os
import logging
import torch
import math
import numpy as np
import glob

def numparameters(model):
  npars = 0 #pars
  nbytes = 0 #bytes
  for name, param in model.named_parameters():
    if param.requires_grad: #learnable parameters only
      npars += param.numel()
      nbytes += param.numel() * param.data.element_size() #returns size of each parameter
      logging.debug("{} => {} = {} x {} bytes".format(name, list(param.data.size()), param.data.numel(), param.data.element_size()))

  name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
  if nbytes == 0:
    i = 0
  else:
    i = int(math.floor(math.log(nbytes, 1024)))
    p = math.pow(1024, i)
    nbytes /= p
  size = "{:.2f}{}".format(nbytes, name[i])

  return npars, size

def average_models(suffix):
  model_files = sorted(glob.glob("{}.checkpoint_????????.pt".format(suffix))) ### I check if there is one model
  if len(model_files) == 0:
    logging.error('No checkpoint found')
    sys.exit()

  avg_model = None
  final_step = 0
  for i, model_file in enumerate(model_files):
    m = torch.load(model_file, map_location='cpu')
    model = m['model']
    step = m['step']
    logging.info('Loading checkpoint step={} file={}'.format(step,model_file))
    if step > final_step:
      final_step = step 

    if i == 0:
      avg_model = model
    else:
      for (k, v) in avg_model.items():
        avg_model[k].mul_(i).add_(model[k]).div_(i + 1)

  final = {"model": avg_model, "step": final_step}
  torch.save(final, "{}.checkpoint_{:08d}_average.pt".format(suffix,final_step))


def save_checkpoint(suffix, model, optimizer, step, keep_last_n):
  checkpoint = { 'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict() }
  torch.save(checkpoint, "{}.checkpoint_{:08d}.pt".format(suffix,step))
  logging.info('Saved {}.checkpoint_{:08d}.pt'.format(suffix,step))
  files = sorted(glob.glob(suffix + '.checkpoint_????????.pt')) 
  while keep_last_n > 0 and len(files) > keep_last_n:
    f = files.pop(0)
    os.remove(f) ### first is the oldest
    logging.debug('Removed checkpoint {}'.format(f))

def load_checkpoint(suffix, model, device, fmodel=None):
  step = 0
  if fmodel is not None:
    if not os.path.isfile(fmodel):
      logging.error('No model found')
      sys.exit()
    file = fmodel

  else:
    files = sorted(glob.glob("{}.checkpoint_????????.pt".format(suffix))) ### I check if there is one model
    if len(files) == 0:
      logging.error('No checkpoint found')
      sys.exit()
    file = files[-1] ### last is the newest

  logging.info('Loading checkpoint file={}'.format(file))
  checkpoint = torch.load(file, map_location=device)
  step = checkpoint['step']
  ### assert checkpoint['model'] has same options than model
  model.load_state_dict(checkpoint['model'])
  logging.info('Loaded model step={} from {}'.format(step,file))
  return step, model

def load_checkpoint_or_initialise(suffix, model, optimizer, device):
  step = 0
  files = sorted(glob.glob("{}.checkpoint_????????.pt".format(suffix))) ### I check if there is one model
  if len(files) == 0:
    for p in model.parameters():
      if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    logging.info('No model found [network initialised]')
    return step, model, optimizer

  file = files[-1] ### last is the newest
  logging.info('Loading checkpoint file={}'.format(file))
  checkpoint = torch.load(file, map_location=device)
  step = checkpoint['step']
  ### assert checkpoint['model'] has same options than model
  model.load_state_dict(checkpoint['model'])
  if optimizer is None:
    logging.info('Loaded model step={} from {}'.format(step,file))
    return step, model, optimizer ### this is for inference

  optimizer.load_state_dict(checkpoint['optimizer'])
  logging.info('Loaded model/optimizer step={} from {}'.format(step,file))
  return step, model, optimizer ### this is for learning

def prepare_source(batch_src, idx_pad, device):
  src = [torch.tensor(seq) for seq in batch_src] #[bs, ls]
  src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=idx_pad).to(device) #[bs,ls]
  msk_src = (src != idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
  if False:
    print('batch')
    for i in range(len(src)):
      print('src[{}]: '.format(i) + ' '.join(['{: ^5}'.format(t) for t in src[i].tolist()]))
      print('msk[{}]: '.format(i) + ' '.join(['{: ^5}'.format(t) for t in msk_src[i,0].tolist()]))
  return src, msk_src

def prepare_target(batch_tgt, idx_pad, device):
  tgt = [torch.tensor(seq[:-1]) for seq in batch_tgt] #delete <eos>
  tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=idx_pad).to(device) 
  ref = [torch.tensor(seq[1:])  for seq in batch_tgt] #delete <bos>
  ref = torch.nn.utils.rnn.pad_sequence(ref, batch_first=True, padding_value=idx_pad).to(device)
  msk_tgt = (tgt != idx_pad).unsqueeze(-2) & (1 - torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)).bool() #[bs,lt,lt]
  if False:
    for i in range(len(tgt)):
      print('tgt[{}]: '.format(i) + ' '.join(['{: ^5}'.format(t) for t in tgt[i].tolist()]))
      print('ref[{}]: '.format(i) + ' '.join(['{: ^5}'.format(t) for t in ref[i].tolist()]))
      for j in range(len(msk_tgt[i])):
        print('msk[{}]: '.format(i) + ' '.join(['{: ^5}'.format(t) for t in msk_tgt[i,j].tolist()]))
  return tgt, ref, msk_tgt

##############################################################################################################
### Endcoder_Decoder #########################################################################################
##############################################################################################################
class Encoder_Decoder(torch.nn.Module):
  #https://www.linzehui.me/images/16005200579239.jpg
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, share_embeddings, src_voc_size, tgt_voc_size, idx_pad):
    super(Encoder_Decoder, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = Embedding(src_voc_size, emb_dim, idx_pad) 
    self.tgt_emb = Embedding(tgt_voc_size, emb_dim, idx_pad) 
    if share_embeddings:
      self.tgt_emb.emb.weight = self.src_emb.emb.weight

    self.add_pos_enc = AddPositionalEncoding(emb_dim, dropout, max_len=5000)
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.stacked_decoder = Stacked_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.generator = Generator(emb_dim, tgt_voc_size)

  def forward(self, src, tgt, msk_src, msk_tgt):
    #src is [bs,ls]
    #tgt is [bs,lt]
    #msk_src is [bs,1,ls] (False where <pad> True otherwise)
    #mst_tgt is [bs,lt,lt]
    ### encoder #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    ### decoder #####
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]
    ### generator ###
    y = self.generator(z_tgt) #[bs, lt, Vt]
    return y ### returns logits (for learning)

  def encode(self, src, msk_src):
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    return z_src

  def decode(self, z_src, tgt, msk_src, msk_tgt=None):
    assert z_src.shape[0] == tgt.shape[0] ### src/tgt batch_sizes must be equal
    #z_src are the embeddings of the source words (encoder) [bs, sl, ed]
    #tgt is the history (words already generated) for current step [bs, lt]
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]
    y = self.generator(z_tgt) #[bs, lt, Vt]
    y = torch.nn.functional.log_softmax(y, dim=-1) 
    return y ### returns log_probs (for inference)

##############################################################################################################
### Embedding ################################################################################################
##############################################################################################################
class Embedding(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, idx_pad):
    super(Embedding, self).__init__()
    self.emb = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=idx_pad)
    self.sqrt_emb_dim = math.sqrt(emb_dim)

  def forward(self, x):
    return self.emb(x) * self.sqrt_emb_dim

##############################################################################################################
### PositionalEncoding #######################################################################################
##############################################################################################################
class AddPositionalEncoding(torch.nn.Module):
  def __init__(self, emb_dim, dropout, max_len=1000):
    super(AddPositionalEncoding, self).__init__()
    assert emb_dim%2 == 0, 'emb_dim must be pair'
    self.dropout = torch.nn.Dropout(dropout)

    position = torch.arange(0, max_len).unsqueeze(1) #[max_len, 1]
    div_term = torch.exp((torch.arange(0, emb_dim, 2, dtype=torch.float) * -(math.log(10000.0) / emb_dim))) #[ed/2]

    pe = torch.zeros(max_len, emb_dim) #[max_len, ed]
    pe[:, 0::2] = torch.sin(position.float()*div_term) #[max_len, 1] * [1, ed/2] => [max_len, ed] (sets pairs of pe)
    pe[:, 1::2] = torch.cos(position.float()*div_term) #[max_len, 1] * [1, ed/2] => [max_len, ed] (sets odds of pe)
    pe = pe.unsqueeze(0) #[1, max_len=5000, ed]

    self.register_buffer('pe', pe) #register_buffer is for params which are saved&restored in state_dict but not trained 

  def forward(self, x):
    bs, l, ed = x.shape
    x = x + self.pe[:, :l, :] #[bs, l, ed] + [1, l, ed] => [bs, l, ed]
    return self.dropout(x)

##############################################################################################################
### Stacked_Encoder ##########################################################################################
##############################################################################################################
class Stacked_Encoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Encoder, self).__init__()
    self.encoderlayers = torch.nn.ModuleList([Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])
    self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, src, msk):
    for i,encoderlayer in enumerate(self.encoderlayers):
      src = encoderlayer(src, msk) #[bs, ls, ed]
    return self.norm(src)

##############################################################################################################
### Stacked_Decoder ##########################################################################################
##############################################################################################################
class Stacked_Decoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Decoder, self).__init__()
    self.decoderlayers = torch.nn.ModuleList([Decoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])
    self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, tgt, msk_src, msk_tgt):
    for i,decoderlayer in enumerate(self.decoderlayers):
      tgt = decoderlayer(z_src, tgt, msk_src, msk_tgt)
    return self.norm(tgt)

##############################################################################################################
### Encoder ##################################################################################################
##############################################################################################################
class Encoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Encoder, self).__init__()
    self.multihead_attn = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.norm_att = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_ff = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, src, msk):
    #NORM
    tmp1 = self.norm_att(src)
    #ATTN over source words 
    tmp2 = self.multihead_attn(q=tmp1, k=tmp1, v=tmp1, msk=msk) #[bs, ls, ed] contains dropout
    #ADD
    tmp = tmp2 + src

    #NORM
    tmp1 = self.norm_ff(tmp)
    #FF
    tmp2 = self.feedforward(tmp1) #[bs, ls, ed] contains dropout
    #ADD
    z = tmp2 + tmp
    return z

##############################################################################################################
### Decoder ##################################################################################################
##############################################################################################################
class Decoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Decoder, self).__init__()
    self.multihead_attn_self = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.multihead_attn_enc = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.norm_att_self = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_att_enc = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_ff = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, tgt, msk_src, msk_tgt):
    #NORM
    tmp1 = self.norm_att_self(tgt)
    #ATTN over tgt (previous) words : q, k, v are tgt words
    tmp2 = self.multihead_attn_self(q=tmp1, k=tmp1, v=tmp1, msk=msk_tgt) #[bs, lt, ed] contains dropout
    #ADD
    tmp = tmp2 + tgt 

    #NORM
    tmp1 = self.norm_att_enc(tmp)
    #ATTN over src words : q are tgt words, k, v are src words
    tmp2 = self.multihead_attn_enc(q=tmp1, k=z_src, v=z_src, msk=msk_src) #[bs, lt, ed] contains dropout
    #ADD
    tmp = tmp2 + tmp

    #NORM
    tmp1 = self.norm_ff(tmp)
    #FF
    tmp2 = self.feedforward(tmp1) #[bs, lt, ed] contains dropout
    #ADD
    z = tmp2 + tmp
    return z

##############################################################################################################
### MultiHead_Attn ###########################################################################################
##############################################################################################################
class MultiHead_Attn(torch.nn.Module):
  def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(MultiHead_Attn, self).__init__()
    self.nh = n_heads
    self.ed = emb_dim
    self.qd = qk_dim
    self.kd = qk_dim
    self.vd = v_dim
    self.WQ = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WK = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WV = torch.nn.Linear(emb_dim, v_dim*n_heads)
    self.WO = torch.nn.Linear(v_dim*n_heads, emb_dim)
    self.dropout = torch.nn.Dropout(dropout)
    self.softmax = torch.nn.Softmax(dim=-1)

  def forward(self, q, k, v, msk=None):
    #q is [bs, lq, ed]
    #k is [bs, lk, ed]
    #v is [bs, lv, ed]
    #msk is [bs, 1, ls] or [bs, lt, lt]
    if msk is not None:
      msk = msk.unsqueeze(1) #[bs, 1, 1, ls] or [bs, 1, lt, lt]
    bs = q.shape[0]
    lq = q.shape[1] ### sequence length of q vectors (length of target sentences)
    lk = k.shape[1] ### sequence length of k vectors (may be length of source/target sentences)
    lv = v.shape[1] ### sequence length of v vectors (may be length of source/target sentences)
    ed = q.shape[2]
    assert self.ed == q.shape[2] == k.shape[2] == v.shape[2]
    assert lk == lv #when applied in decoder both refer the source-side (lq refers the target-side)
    Q = self.WQ(q).contiguous().view([bs,lq,self.nh,self.qd]).permute(0,2,1,3) #=> [bs,lq,nh*qd] => [bs,lq,nh,qd] => [bs,nh,lq,qd]
    K = self.WK(k).contiguous().view([bs,lk,self.nh,self.kd]).permute(0,2,1,3) #=> [bs,lk,nh*kd] => [bs,lk,nh,kd] => [bs,nh,lk,kd]
    V = self.WV(v).contiguous().view([bs,lv,self.nh,self.vd]).permute(0,2,1,3) #=> [bs,lv,nh*vd] => [bs,lv,nh,vd] => [bs,nh,lv,vd]
    #Scaled dot-product Attn from multiple Q, K, V vectors (bs*nh*l vectors)
    Q = Q / math.sqrt(self.kd)
    s = torch.matmul(Q, K.transpose(2, 3)) #[bs,nh,lq,qd] x [bs,nh,kd,lk] = [bs,nh,lq,lk] # thanks to qd==kd #in decoder lq are target words and lk are source words
    if msk is not None:
      s = s.masked_fill(msk == 0, float('-inf')) #score=-Inf to masked tokens
    w = self.softmax(s) #[bs,nh,lq,lk] (these are the attention weights)
    w = self.dropout(w) #[bs,nh,lq,lk] 

    z = torch.matmul(w,V) #[bs,nh,lq,lk] x [bs,nh,lv,vd] = [bs,nh,lq,vd] #thanks to lk==lv
    z = z.transpose(1,2).contiguous().view([bs,lq,self.nh*self.vd]) #=> [bs,lq,nh,vd] => [bs,lq,nh*vd]
    z = self.WO(z) #[bs,lq,ed]
    return self.dropout(z)

##############################################################################################################
### FeedForward ##############################################################################################
##############################################################################################################
class FeedForward(torch.nn.Module):
  def __init__(self, emb_dim, ff_dim, dropout):
    super(FeedForward, self).__init__()
    self.FF_in = torch.nn.Linear(emb_dim, ff_dim)
    self.FF_out = torch.nn.Linear(ff_dim, emb_dim)
    self.dropout = torch.nn.Dropout(dropout) #this regularization is not used in the original model

  def forward(self, x):
    tmp = self.FF_in(x)
    tmp = torch.nn.functional.relu(tmp)
    tmp = self.dropout(tmp)
    tmp = self.FF_out(tmp)
    tmp = self.dropout(tmp)
    return tmp

##############################################################################################################
### Generator ################################################################################################
##############################################################################################################
class Generator(torch.nn.Module):
  def __init__(self, emb_dim, voc_size):
    super(Generator, self).__init__()
    self.proj = torch.nn.Linear(emb_dim, voc_size) #[bs, Vt]

  def forward(self, x):
    y = self.proj(x)
    return y



