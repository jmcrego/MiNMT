# -*- coding: utf-8 -*-

import sys
import os
import logging
import torch
import math
import numpy as np
import glob
from transformer.Model import Embedding, AddPositionalEncoding, Stacked_Encoder, Stacked_Decoder, Encoder, Decoder, MultiHead_Attn, FeedForward, Generator

##############################################################################################################
### Encoder_Decoder_scc ######################################################################################
##############################################################################################################
class Encoder_Decoder_scc(torch.nn.Module):
  #https://www.linzehui.me/images/16005200579239.jpg
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, share_embeddings, src_voc_size, tgt_voc_size, idx_pad):
    super(Encoder_Decoder_scc, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = Embedding(src_voc_size, emb_dim, idx_pad) 
    self.tgt_emb = Embedding(tgt_voc_size, emb_dim, idx_pad) 
    if share_embeddings:
      self.tgt_emb.emb.weight = self.src_emb.emb.weight

    self.add_pos_enc = AddPositionalEncoding(emb_dim, dropout, max_len=5000) 
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)         ### encoder for src and xsrc
    self.stacked_encoder_scc = Stacked_Encoder_scc(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) ### encoder for xtgt
    self.stacked_decoder_scc = Stacked_Decoder_scc(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) ### decoder for tgt
    self.generator_msk = Generator(emb_dim, tgt_voc_size)
    self.generator_trn = Generator(emb_dim, tgt_voc_size)

  def type(self):
    return 'scc'

  def forward(self, src, xsrc, xtgt, tgt, msk_src, msk_xsrc, msk_xtgt, msk_tgt): 
    #src is [bs,ls]
    #tgt is [bs,lt]
    #msk_src is [bs,1,ls] (False where <pad> True otherwise)
    #mst_tgt is [bs,lt,lt]

    ### encoder #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,ls,ed]
    z_xtgt = self.stacked_encoder_scc(z_src, z_xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt) #[bs,ls,ed]
    ### generator ###
    y_msk = self.generator_msk(z_xtgt) #[bs, lt, Vt]

    ### decoder #####
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder_scc(z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt) #[bs,lt,ed]
    ### generator ###
    y_trn = self.generator_trn(z_tgt) #[bs, lt, Vt]

    return y_msk, y_trn ### returns logits (for learning)

  def encode(self, src, xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt):
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,ls,ed]
    z_xtgt = self.stacked_encoder_scc(z_src, z_xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt) #[bs,ls,ed]
    return z_src, z_xtgt

  def decode(self, z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt=None):
    assert z_src.shape[0] == tgt.shape[0] ### src/tgt batch_sizes must be equal
    #z_src are the embeddings of the source words (encoder) [bs, sl, ed]
    #tgt is the history (words already generated) for current step [bs, lt]
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder_scc(z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt) #[bs,lt,ed]
    ### generator ###
    y = self.generator_trn(z_tgt) #[bs, lt, Vt]
    y = torch.nn.functional.log_softmax(y, dim=-1) 
    return y ### returns log_probs (for inference)

##############################################################################################################
### Stacked_Encoder_scc ######################################################################################
##############################################################################################################
class Stacked_Encoder_scc(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Encoder_scc, self).__init__()
    self.encoderlayers = torch.nn.ModuleList([Encoder_scc(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])
    self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, z_xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt):
    for i,encoderlayer in enumerate(self.encoderlayers):
      xtgt = encoderlayer(z_src, z_xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt) #[bs, ls, ed]
    return self.norm(xtgt)

##############################################################################################################
### Stacked_Decoder_scc ######################################################################################
##############################################################################################################
class Stacked_Decoder_scc(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Stacked_Decoder_scc, self).__init__()
    self.decoderlayers = torch.nn.ModuleList([Decoder_scc(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])
    self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt):
    for i,decoderlayer in enumerate(self.decoderlayers):
      tgt = decoderlayer(z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt)
    return self.norm(tgt)

##############################################################################################################
### Encoder_scc ##############################################################################################
##############################################################################################################
class Encoder_scc(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Encoder_scc, self).__init__()
    self.multihead_attn_self = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.multihead_attn_cross1 = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.multihead_attn_cross2 = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.norm_att_self = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_att_cross1 = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_att_cross2 = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_ff = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, z_xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt):
    #NORM
    tmp_norm = self.norm_att_self(xtgt)
    #Self ATTN over xtgt words 
    tmp2 = self.multihead_attn_self(q=tmp_norm, k=tmp_norm, v=tmp_norm, msk=msk_xtgt) #[bs, ls, ed] contains dropout
    #ADD
    tmp = tmp2 + xtgt

    #NORM
    tmp_norm = self.norm_att_cross1(tmp)
    #Cross ATTN over xsrc words : q are xtgt words, k, v are xsrc words
    tmp2 = self.multihead_attn_cross1(q=tmp_norm, k=z_xsrc, v=z_xsrc, msk=msk_xsrc) #[bs, ls, ed] contains dropout
    #ADD
    tmp = tmp2 + tmp

    #NORM
    tmp_norm = self.norm_att_cross2(tmp)
    #Cross ATTN over src words : q are xtgt words, k, v are src words
    tmp2 = self.multihead_attn_cross2(q=tmp_norm, k=z_src, v=z_src, msk=msk_src) #[bs, ls, ed] contains dropout
    #ADD
    tmp = tmp2 + tmp

    #NORM
    tmp_norm = self.norm_ff(tmp)
    #FF
    tmp2 = self.feedforward(tmp_norm) #[bs, ls, ed] contains dropout
    #ADD
    z = tmp2 + tmp
    return z

##############################################################################################################
### Decoder_scc ##############################################################################################
##############################################################################################################
class Decoder_scc(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Decoder_scc, self).__init__()
    self.multihead_attn_self = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.multihead_attn_cross1 = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.multihead_attn_cross2 = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.norm_att_self = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_att_cross1 = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_att_cross2 = torch.nn.LayerNorm(emb_dim, eps=1e-6) 
    self.norm_ff = torch.nn.LayerNorm(emb_dim, eps=1e-6) 

  def forward(self, z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt):
    #NORM
    tmp_norm = self.norm_att_self(tgt)
    #Self ATTN over tgt (previous) words : q, k, v are tgt words
    tmp2 = self.multihead_attn_self(q=tmp_norm, k=tmp_norm, v=tmp_norm, msk=msk_tgt) #[bs, lt, ed] contains dropout
    #ADD
    tmp = tmp2 + tgt 

    #NORM
    tmp_norm = self.norm_att_cross1(tmp)
    #Cross ATTN over xtgt words : q are tgt words, k, v are xtgt words
    tmp2 = self.multihead_attn_cross1(q=tmp_norm, k=z_xtgt, v=z_xtgt, msk=msk_xtgt) #[bs, lt, ed] contains dropout
    #ADD
    tmp = tmp2 + tmp

    #NORM
    tmp_norm = self.norm_att_cross2(tmp)
    #Cross ATTN over src words : q are tgt words, k, v are src words
    tmp2 = self.multihead_attn_cross2(q=tmp_norm, k=z_src, v=z_src, msk=msk_src) #[bs, lt, ed] contains dropout
    #ADD
    tmp = tmp2 + tmp

    #NORM
    tmp_norm = self.norm_ff(tmp)
    #FF
    tmp2 = self.feedforward(tmp_norm) #[bs, lt, ed] contains dropout
    #ADD
    z = tmp2 + tmp
    return z


