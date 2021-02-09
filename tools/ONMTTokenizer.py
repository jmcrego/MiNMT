# -*- coding: utf-8 -*-

import sys
import pyonmttok

class ONMTTokenizer():
	def __init__(self, sp_model=None):
		if sp_model is None or not os.path.exists(sp_model):
			self.tokenizer = pyonmttok.Tokenizer(mode = 'space', spacer_annotate = True)
		else:
			self.tokenizer = pyonmttok.Tokenizer(mode = 'none', spacer_annotate = True, sp_model_path = sp_model)

	def tokenize(self, text):
		return self.tokenizer.tokenize(text)[0]

	def detokenize(self, tokens):
		return self.tokenizer.detokenize(tokens)
