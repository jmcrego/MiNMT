# -*- coding: utf-8 -*-

import sys
import logging
import torch
import math
#import numpy as np

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

def build_model(o,src_vocab,tgt_vocab):
  m = Model_encoder_decoder(o.network.n_layers, o.network.ff_dim, o.network.n_heads, o.network.emb_dim, o.network.qk_dim, o.network.v_dim, len(src_vocab), len(tgt_vocab), src_vocab.idx_pad, o.optim.dropout)
  npars, size = numparameters(m)
  logging.info('Built model #params = {} ({})'.format(npars,size))

  for p in m.parameters():
    if p.dim() > 1:
      torch.nn.init.xavier_uniform_(p)
  logging.info('Model initialised (Xavier uniform)')

  return m

##############################################################################################################
### Model_endocder_decoder ###################################################################################
##############################################################################################################
class Model_encoder_decoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, src_voc_size, tgt_voc_size, idx_pad, dropout): 
    super(Model_encoder_decoder, self).__init__()
    self.src_emb = torch.nn.Embedding(src_voc_size, emb_dim, padding_idx=idx_pad)
    self.tgt_emb = torch.nn.Embedding(tgt_voc_size, emb_dim, padding_idx=idx_pad)
    self.pos_enc = PositionalEncoding(emb_dim, dropout, max_len=5000)
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.stacked_decoder = Stacked_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.generator = Generator(emb_dim, tgt_voc_size)

  def forward(self, src, ref, mask):
    src = self.pos_enc(self.src_emb(src))
    ref = self.pos_enc(self.tgt_emb(ref))
    z_src = self.stacked_encoder(src)
    z_ref = self.stacked_decoder(z_src, ref, mask)
    y = self.generator(z_ref)
    return y

##############################################################################################################
### Stacked_Encoder ##########################################################################################
##############################################################################################################
class Stacked_Encoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Encoder, self).__init__()
    self.encoderlayers = torch.nn.ModuleList([Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])

  def forward(self, x):
    for encoderlayer in self.encoderlayers:
      x = encoderlayer(x)
    return x

##############################################################################################################
### Encoder ##################################################################################################
##############################################################################################################
class Encoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Encoder, self).__init__()
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.multiheaded_selfattn = MultiHeaded_SelfAttn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.addnorm = AddNorm(emb_dim) 
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, src):
    tmp = self.dropout(self.addnorm(self.multiheaded_selfattn(q=src, k_and_v=src), src))
    z = self.dropout(self.addnorm(self.feedforward(tmp, mask), tmp))
    return z

##############################################################################################################
### Stacked_Decoder ##########################################################################################
##############################################################################################################
class Stacked_Decoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Decoder, self).__init__()
    self.decoderlayers = torch.nn.ModuleList([Decoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])

  def forward(self, z_src, x, mask):
    for decoderlayer in self.decoderlayers:
      x = decoderlayer(z_src, x, mask)
    return x

##############################################################################################################
### Decoder ##################################################################################################
##############################################################################################################
class Decoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Decoder, self).__init__()
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.multiheaded_selfattn = MultiHeaded_SelfAttn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.addnorm = AddNorm(emb_dim) 
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, z_src, ref, mask):
    tmp = self.dropout(self.addnorm(self.multiheaded_selfattn(q=ref, k_and_v=ref, mask=mask), ref)) 
    tmp = self.dropout(self.addnorm(self.multiheaded_selfattn(q=tmp, k_and_v=z_src), tmp))
    z = self.dropout(self.addnorm(self.feedforward(tmp, mask), tmp))
    return z

##############################################################################################################
### MultiHeaded_SelfAttn #####################################################################################
##############################################################################################################
class MultiHeaded_SelfAttn(torch.nn.Module):
  def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(MultiHeaded_SelfAttn, self).__init__()
    self.WO = torch.nn.Linear(n_heads*v_dim, emb_dim)
    self.attnheads = torch.nn.ModuleList([SelfAttn(emb_dim, qk_dim, v_dim, dropout) for _ in range(n_heads)])
    self.dropout = torch.nn.Dropout(dropout)

  def forward(q, k_and_v, mask=None):
    z = torch.cat([attnhead(q, k_and_v, mask) for attnhead in self.attnheads], dim=2) 
    return self.dropout(self.WO(z))

##############################################################################################################
### SelfAttn #################################################################################################
##############################################################################################################
class SelfAttn(torch.nn.Module):
  def __init__(self, emb_dim, qk_dim, v_dim, dropout):
    super(SelfAttn, self).__init__()
    self.WQ = torch.nn.Linear(emb_dim, qk_dim)
    self.WK = torch.nn.Linear(emb_dim, qk_dim)
    self.WV = torch.nn.Linear(emb_dim, v_dim)

  def forward(q, k_and_v, mask=None):  ### implement future masking if mask is not None
    q = self.WQ(q)
    k = self.WK(k_and_v)
    v = self.WV(k_and_v)
    ### implements scaled dot-product attention for q, k, v
    s = q.bmm(k.transpose(1, 2))
    w = torch.nn.functional.softmax(s / qk_dim**0.5, dim=-1)
    z = w.bmm(v)
    return z

##############################################################################################################
### FeedForward ##############################################################################################
##############################################################################################################
class FeedForward(torch.nn.Module):
  def __init__(self, emb_dim, ff_dim, dropout):
    super(FeedForward, self).__init__()
    self.FF_in = torch.nn.Linear(emb_dim, ff_dim)
    self.FF_out = torch.nn.Linear(ff_dim, emb_dim)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x):
    return self.FF_out(self.dropout(torch.nn.functional.relu(self.FF_in(x))))

##############################################################################################################
### AddNorm ##################################################################################################
##############################################################################################################
class AddNorm(torch.nn.Module):
  #implements adding residual connections & normalization
  def __init__(self, emb_dim):
    super(AddNorm, self).__init__()
    self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6)

  def forward(a, b, mask):
    return self.norm(a + b) ### b is the residual

##############################################################################################################
### PositionalEncoding #######################################################################################
##############################################################################################################
class PositionalEncoding(torch.nn.Module):
  def __init__(self, emb_dim, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = torch.nn.Dropout(dropout)
    pe = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe) #register_buffer are for params saved&restored in state_dict not trained 

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

##############################################################################################################
### Generator ################################################################################################
##############################################################################################################
class Generator(torch.nn.Module):
  def __init__(self, emb_dim, voc_size):
    super(Generator, self).__init__()
    self.proj = torch.nn.Linear(emb_dim, voc_size)

  def forward(self, x):
    return torch.nn.functional.log_softmax(self.proj(x), dim=-1)



