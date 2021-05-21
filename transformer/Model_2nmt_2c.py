# -*- coding: utf-8 -*-

import sys
import os
import logging
import torch
import math
import numpy as np
import glob
from transformer.Model import Embedding, AddPositionalEncoding, Stacked_Encoder, Stacked_Decoder, Encoder, Decoder, Stacked_Encoder_scc, Stacked_Decoder_scc, Encoder_scc, Decoder_scc, Stacked_CrossAdapter, MultiHead_Attn, FeedForward, Generator

##############################################################################################################
### Encoder_Decoder_s_s_scc_scc ######################################################################################
##############################################################################################################
class Encoder_Decoder_2nmt_2c(torch.nn.Module):
  #https://www.linzehui.me/images/16005200579239.jpg
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, share_embeddings, src_voc_size, tgt_voc_size, idx_pad):
    super(Encoder_Decoder_2nmt_2c, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = Embedding(src_voc_size, emb_dim, idx_pad)
    self.tgt_emb = Embedding(tgt_voc_size, emb_dim, idx_pad)
    if share_embeddings:
        self.tgt_emb.emb.weight = self.src_emb.emb.weight

    self.add_pos_enc = AddPositionalEncoding(emb_dim, dropout, max_len=5000)
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.stacked_decoder = Stacked_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.cross_xtgt = Stacked_CrossAdapter(1, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.cross_tgt = Stacked_CrossAdapter(1, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)

    self.generator_hide = Generator(emb_dim, tgt_voc_size)
    self.generator_trns = Generator(emb_dim, tgt_voc_size)

  def type(self):
    return '2nmt_2c'

  def forward(self, src, xsrc, xtgt, tgt, msk_src, msk_xsrc, msk_xtgt, msk_tgt, msk_tgt_cross): 
    #src is [bs,ls]
    #tgt is [bs,lt]
    #msk_src is [bs,1,ls] (False where <pad> True otherwise)
    #mst_tgt is [bs,lt,lt]

    ### encoder_xsrc #####
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]
    ### decoder_xtgt #####
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,lt,ed]
    z_xtgt = self.stacked_decoder(z_xsrc, xtgt, msk_xsrc, msk_xtgt) #[bs,lt,ed]

    ### encoder_src #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    ### decoder_tgt #####
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]

    ### cross_xtgt ###
    z_xtgt = self.cross_xtgt(z_src, z_xtgt, msk_src, msk_xtgt)
    ### cross_tgt ###
    z_tgt = self.cross_tgt(z_xtgt, z_tgt, msk_xtgt, msk_tgt_cross)

    ### generator_hide ###
    y_hide = self.generator_hide(z_xtgt) #[bs, lt, Vt]
    ### generator_trns ###
    y_trns = self.generator_trns(z_tgt) #[bs, lt, Vt]

    return y_hide, y_trns ### returns logits (for learning)

  def encode(self, src, xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt):
    ### encoder_xsrc #####
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]

    ### decoder_xtgt #####
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,lt,ed]
    z_xtgt = self.stacked_decoder(z_xsrc, xtgt, msk_xsrc, msk_xtgt) #[bs,lt,ed]

    ### encoder_src #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]

    ### cross_xtgt ###
    z_xtgt = self.cross_xtgt(z_xtgt, z_src, msk_xtgt, msk_xsrc)

    ### generator_hide ###
    #y_hide = self.generator_hide(z_xtgt) #[bs, lt, Vt]
    #y_hide = torch.nn.functional.log_softmax(y_hide, dim=-1) 

    return z_src, z_xtgt

  def decode(self, z_src, z_xtgt, tgt, msk_src, msk_xtgt, msk_tgt=None):
    assert z_src.shape[0] == tgt.shape[0] ### src/tgt batch_sizes must be equal
    #z_src are the embeddings of the source words (encoder) [bs, sl, ed]
    #tgt is the history (words already generated) for current step [bs, lt]

    ### decoder_tgt #####
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]

    ### cross_tgt ###
    z_tgt = self.cross_tgt(z_tgt, z_xtgt, msk_tgt, msk_xtgt)

    ### generator_trns ###
    y_trns = self.generator_trns(z_tgt) #[bs, lt, Vt]
    y_trns = torch.nn.functional.log_softmax(y_trns, dim=-1) 

    return y_trns ### returns log_probs (for inference)



