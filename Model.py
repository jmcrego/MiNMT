# -*- coding: utf-8 -*-

import sys
import logging
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import copy, #math, time

def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


##############################################################################################################
### Stacked_Encoder ##########################################################################################
##############################################################################################################
class Stacked_Encoder(nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    self.encoderlayers = clones(Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout), n_layers)

  def forward(self, x, mask):
    for encoderlayer in self.encoderlayers:
      x = encoderlayer(x, mask)
    return x

##############################################################################################################
### Encoder ##################################################################################################
##############################################################################################################
class Encoder(nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    self.multiheaded_selfattn = MultiHeaded_SelfAttn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.addnorm = AddNorm(emb_dim, dropout)

  def forward(self, x, mask):
    x = self.addnorm(self.multiheaded_selfattn(x, mask), x, mask)
    return self.addnorm(self.feedforward(x, mask), x, mask)

##############################################################################################################
### MultiHeaded_SelfAttn #####################################################################################
##############################################################################################################
class MultiHeaded_SelfAttn(nn.Module):
  def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
    self.heads = clones(SelfAttn(emb_dim, qk_dim, v_dim, dropout), n_heads)
    self.WO = nn.Linear(n_heads*v_dim, emb_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(x, mask):
    z = torch.cat([head(x, mask) for head in self.heads], dim=2) 
    return self.dropout(self.WO(z))

##############################################################################################################
### SelfAttn #################################################################################################
##############################################################################################################
class SelfAttn(nn.Module):
  def __init__(self, emb_dim, qk_dim, v_dim, dropout):
    self.WQ = nn.Linear(emb_dim, qk_dim)
    self.WK = nn.Linear(emb_dim, qk_dim)
    self.WV = nn.Linear(emb_dim, v_dim)

  def forward(x, mask):
    q = self.WQ(x)
    k = self.WK(x)
    v = self.WV(x)
    ### implements scaled dot-product attention for q, k, v
    s = q.bmm(k.transpose(1, 2))
    w = F.softmax(s / qk_dim**0.5, dim=-1)
    z = w.bmm(v)
    return z

##############################################################################################################
### FeedForward ##############################################################################################
##############################################################################################################
class FeedForward(nn.Module):
  def __init__(self, emb_dim, ff_dim, dropout):
    self.ff_in = nn.Linear(emb_dim, ff_dim)
    self.ff_out = nn.Linear(ff_dim, emb_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.ff_out(self.dropout(F.relu(self.ff_in(x))))

##############################################################################################################
### AddNorm ##################################################################################################
##############################################################################################################
class AddNorm(nn.Module):
  def __init__(self, emb_dim, dropout):
    self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(a, b, mask):
    return self.norm(self.dropout(a) + b) ### b is the residual

##############################################################################################################
### PositionalEncoding #######################################################################################
##############################################################################################################
class PositionalEncoding(nn.Module):
  def __init__(self, emb_dim, dropout, max_len=5000):
    self.dropout = nn.Dropout(dropout)
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
class Generator(nn.Module):
  def __init__(self, emb_dim, voc_size):
    self.proj = nn.Linear(emb_dim, voc_size)

  def forward(self, x):
    return F.log_softmax(self.proj(x), dim=-1)


