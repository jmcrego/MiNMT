# -*- coding: utf-8 -*-

import sys
import pyonmttok

class ONMTTokenizer():
  def __init__(self, sp_model=None):
  	if sp_model is not None:
	    self.tokenizer = pyonmttok.Tokenizer(mode = 'space', sp_model_path = sp_model, joiner_annotate = True)
	else:
	    self.tokenizer = pyonmttok.Tokenizer(mode = 'space', joiner_annotate = True)

  def tokenize(self, text):
    return self.tokenizer.tokenize(text)[0]

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)
