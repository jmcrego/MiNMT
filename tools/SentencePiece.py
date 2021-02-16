# -*- coding: utf-8 -*-
import sys
import logging
import sentencepiece as spm

def fd2list(fin, type=None):
	### read file
	if fin is None:
	 	fd = sys.stdin
	else:
		fd = open(fin,'r')
	if type is None:
		lines = fd.read().splitlines()
	elif type == 'int':
		lines = [[int(s) for s in l.split()] for l in fd.read().splitlines()]
	elif type == 'str':
		lines = [[s for s in l.split()] for l in fd.read().splitlines()]
	else:
		logging.error('Invalid type {}'.format(type))
		sys.exit()
	if fin is not None:
		fd.close()
	return lines

class SentencePiece():
	def __init__(self, sp_model=None):
		self.sp = None
		self.idx_pad = 0 
		self.str_pad = '<pad>'
		self.idx_unk = 1 
		self.str_unk = '<unk>'
		self.idx_bos = 2
		self.str_bos = '<bos>'
		self.idx_eos = 3
		self.str_eos = '<eos>'

		if sp_model is not None: 
			self.sp = spm.SentencePieceProcessor(model_file=sp_model)
			assert self.sp.piece_to_id(self.str_pad) == 0, '<pad> must exist in vocab with id=0 while found id={}'.format(self.sp.piece_to_id(self.str_pad))
			assert self.sp.piece_to_id(self.str_unk) == 1, '<unk> must exist in vocab with id=1 while found id={}'.format(self.sp.piece_to_id(self.str_unk))
			assert self.sp.piece_to_id(self.str_bos) == 2, '<bos> must exist in vocab with id=2 while found id={}'.format(self.sp.piece_to_id(self.str_bos))
			assert self.sp.piece_to_id(self.str_eos) == 3, '<eos> must exist in vocab with id=3 while found id={}'.format(self.sp.piece_to_id(self.str_eos))
			logging.info('Read SentencePiece model with {} tokens ~ {}'.format(len(self.sp), sp_model))

	def train(self, sp_model, fins, vocab_size=30000, character_coverage=0.9995, input_sentence_size=1000000, shuffle_input_sentence=True, max_sentence_length=200):
		if sp_model is not None:
			self.sp = spm.SentencePieceTrainer.train(input=','.join(fins), model_prefix=sp_model, vocab_size=vocab_size, character_coverage=character_coverage,	input_sentence_size=input_sentence_size, shuffle_input_sentence=shuffle_input_sentence,	max_sentence_length=max_sentence_length, pad_id=0, pad_piece='<pad>', unk_id=1, unk_piece='<unk>', bos_id=2, bos_piece='<bos>', eos_id=3, eos_piece='<eos>')

	def encode(self, fin, out_type=int):
		raw_lines = fd2list(fin, type=None) #list of strings
		tok_lines = self.sp.encode(raw_lines, out_type=out_type)
		return raw_lines, tok_lines

	def decode(self, fin, in_type=int):
		if isinstance(fin, str):
			tok_lines = fd2list(fin, type=in_type) #list of list of strings_or_ints
		else:
			tok_lines = fin
		raw_lines = self.sp.decode(tok_lines)
		return tok_lines, raw_lines

	def __len__(self):
		return len(self.sp)

	def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
		if type(s) == int: 
			return self.sp.id_to_piece(s) != self.str_unk ### testing an index
		return self.sp.piece_to_id(s) != self.idx_unk ### testing a string

	def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
		if type(s) == int: 
			return self.sp.id_to_piece(s) ### return a string
		return self.sp.piece_to_id(s) ### return an index


