# -*- coding: utf-8 -*-

import sys
import pyonmttok
#import logging

####################################################################
### ONMTTokenizer ##################################################
####################################################################
class ONMTTokenizer():
  def __init__(self, sp_model):
    self.tokenizer = pyonmttok.Tokenizer(mode = None, sp_model_path = sp_model, joiner_annotate = True)

  def tokenize(self, text):
    return self.tokenizer.tokenize(text)[0]

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)

