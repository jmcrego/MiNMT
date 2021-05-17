# -*- coding: utf-8 -*-

import sys
import os
import logging
import torch
import math
import numpy as np
import glob
from transformer.Model import Embedding, AddPositionalEncoding, Stacked_Encoder, Stacked_Decoder, Encoder, Decoder, Stacked_Encoder_scc, Stacked_Decoder_scc, Encoder_scc, Decoder_scc, MultiHead_Attn, FeedForward, Generator

##############################################################################################################
### Encoder_Decoder_s_s_scc_scc ######################################################################################
##############################################################################################################
class Encoder_Decoder_s_s_scc_scc(torch.nn.Module):
  #https://www.linzehui.me/images/16005200579239.jpg
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, share_embeddings, share_encoders, src_voc_size, tgt_voc_size, idx_pad):
    super(Encoder_Decoder_s_s_scc_scc, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = Embedding(src_voc_size, emb_dim, idx_pad) 
    self.tgt_emb = Embedding(tgt_voc_size, emb_dim, idx_pad) 
    if share_embeddings:
        self.tgt_emb.emb.weight = self.src_emb.emb.weight

    self.add_pos_enc = AddPositionalEncoding(emb_dim, dropout, max_len=5000) 
    self.stacked_encoder_s1 = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)         ### encoder for src and xsrc
    self.stacked_encoder_s2 = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)         ### encoder for src and xsrc
    if share_encoders:
        self.stacked_encoder_s2 = self.stacked_encoder_s1
    self.stacked_encoder_scc = Stacked_Encoder_scc(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) ### encoder for xtgt
    self.stacked_decoder_scc = Stacked_Decoder_scc(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) ### decoder for tgt
    self.generator_msk = Generator(emb_dim, tgt_voc_size)
    self.generator_trn = Generator(emb_dim, tgt_voc_size)

  def type(self):
    return 's_s_scc_scc'

  def forward(self, src, xsrc, xtgt, tgt, msk_src, msk_xsrc, msk_xtgt, msk_tgt): 
    #src is [bs,ls]
    #tgt is [bs,lt]
    #msk_src is [bs,1,ls] (False where <pad> True otherwise)
    #mst_tgt is [bs,lt,lt]

    ### encoder #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder_s1(src, msk_src) #[bs,ls,ed]
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder_s2(xsrc, msk_xsrc) #[bs,ls,ed]
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
    z_src = self.stacked_encoder_s1(src, msk_src) #[bs,ls,ed]
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder_s2(xsrc, msk_xsrc) #[bs,ls,ed]
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



