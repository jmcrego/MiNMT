# -*- coding: utf-8 -*-
import sys
import logging
from collections import defaultdict
import sentencepiece as spm

def fd2list(fin, type='str'):
	### whe type is:
	# int: returns list of list of ints (idxs)
	# str: returns list of list of strings (words)

	### open file/stdin
	if fin is None:
	 	fd = sys.stdin
	else:
		fd = open(fin,'r')

	### parse content
	if type == 'int':
		lines = [[int(s) for s in l.split()] for l in fd.read().splitlines()]
	elif type == 'str':
		lines = [[s for s in l.split()] for l in fd.read().splitlines()]
	else:
		logging.error('Invalid type {}'.format(type))
		sys.exit()

	### close file
	if fin is not None:
		fd.close()
	return lines

#######################
### Space tokenizer ###
#######################
class Space():
	def __init__(self, fmod=None):
		self.idx_pad = 0 
		self.str_pad = '<pad>'
		self.idx_unk = 1 
		self.str_unk = '<unk>'
		self.idx_bos = 2
		self.str_bos = '<bos>'
		self.idx_eos = 3
		self.str_eos = '<eos>'
		self.tok_to_idx = defaultdict()
		self.idx_to_tok = []

		if fmod is not None:
			### read vocab from fmod
			with open(fmod,'r') as f: 
				for l in f:
					toks = l.split()
					tok = toks[0]
					if tok in self.tok_to_idx:
						continue
					if ' ' in tok or len(tok) == 0:
						logging.warning('Bad vocab entry: {} [skipping]'.format(tok))
						continue
					self.idx_to_tok.append(tok)
					self.tok_to_idx[tok] = len(self.tok_to_idx)
			logging.debug('Read Vocab ({} entries) from file {}'.format(len(self.idx_to_tok), fmod))
			assert self.tok_to_idx[self.str_pad] == 0, '<pad> must exist in vocab with id=0 while found id={}'.format(self.tok_to_idx[self.str_pad])
			assert self.tok_to_idx[self.str_unk] == 1, '<unk> must exist in vocab with id=1 while found id={}'.format(self.tok_to_idx[self.str_unk])
			assert self.tok_to_idx[self.str_bos] == 2, '<bos> must exist in vocab with id=2 while found id={}'.format(self.tok_to_idx[self.str_bos])
			assert self.tok_to_idx[self.str_eos] == 3, '<eos> must exist in vocab with id=3 while found id={}'.format(self.tok_to_idx[self.str_eos])
			logging.info('Read Space vocab with {} tokens ~ {}'.format(len(self.tok_to_idx), fmod))

	def encode(self, fin, in_type, out_type):
		if in_type == 'int':
			logging.error('Space tokenizer encoder considers only str as in_type')
			sys.exit()
		if out_type == 'str':
			logging.error('Space tokenizer encoder considers only int as out_type')
			sys.exit()

		raw_lines = fd2list(fin, type=in_type) #read into 'str'
		tok_lines = []
		for l in raw_lines:
			tok_line = []
			for tok in l:
				idx = self.tok_to_idx[tok] if tok in self.tok_to_idx else self.idx_unk
				tok_line.append(idx)
			tok_lines.append(tok_line)
		if len(raw_lines) != len(tok_lines):
			logging.info('error: different length raw_lines/tok_lines!')
			sys.exit()
		return raw_lines, tok_lines

	def decode(self, fin, in_type, out_type):
		if in_type == 'str':
			logging.error('Space tokenizer decoder considers only int as in_type')
			sys.exit()
		if out_type == 'int':
			logging.error('Space tokenizer encoder considers only str as out_type')
			sys.exit()

		tok_lines = fd2list(fin, type=in_type) #list of list of strings_or_ints
		raw_lines = []
		for l in tok_lines:
			raw_line = []
			for idx in l:
				tok = self.idx_to_tok[idx] if idx < len(self.idx_to_tok) else self.str_unk
				raw_line.append(tok)
			raw_lines.append(raw_line)
		if len(raw_lines) != len(tok_lines):
			logging.info('error: different length raw_lines/tok_lines!')
			sys.exit()
		return tok_lines, raw_lines

	def decode_list(self, tok_lines):
		tok_lines = map(int,tok_lines)
		raw_line = []
		for idx in tok_lines:
			tok = self.idx_to_tok[idx] if idx < len(self.idx_to_tok) else self.str_unk
			raw_line.append(tok)
		return raw_line

	def __len__(self):
		return len(self.idx_to_tok)

	def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
		if type(s) == int: 
			return s < len(self.idx_to_tok) ### testing an index
		return s in self.tok_to_idx ### testing a string

	def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
		if type(s) == int: ### return a string
			return self.idx_to_tok[s]
		if s in self.tok_to_idx: ### return an index
			return self.tok_to_idx[s] 
		else:
			return self.idx_unk

###############################
### SentencePiece tokenizer ###
###############################
class SentencePiece():
	def __init__(self, fmod=None):
		self.sp = None
		self.idx_pad = 0 
		self.str_pad = '<pad>'
		self.idx_unk = 1 
		self.str_unk = '<unk>'
		self.idx_bos = 2
		self.str_bos = '<bos>'
		self.idx_eos = 3
		self.str_eos = '<eos>'

		if fmod is not None: 
			self.sp = spm.SentencePieceProcessor(model_file=fmod)
			assert self.sp.piece_to_id(self.str_pad) == 0, '<pad> must exist in vocab with id=0 while found id={}'.format(self.sp.piece_to_id(self.str_pad))
			assert self.sp.piece_to_id(self.str_unk) == 1, '<unk> must exist in vocab with id=1 while found id={}'.format(self.sp.piece_to_id(self.str_unk))
			assert self.sp.piece_to_id(self.str_bos) == 2, '<bos> must exist in vocab with id=2 while found id={}'.format(self.sp.piece_to_id(self.str_bos))
			assert self.sp.piece_to_id(self.str_eos) == 3, '<eos> must exist in vocab with id=3 while found id={}'.format(self.sp.piece_to_id(self.str_eos))
			logging.info('Read SentencePiece model with {} tokens ~ {}'.format(len(self.sp), fmod))

	def train(self, fmod, fins, vocab_size=30000, character_coverage=0.9995, input_sentence_size=1000000, shuffle_input_sentence=True, max_sentence_length=200):
		if fmod is not None:
			self.sp = spm.SentencePieceTrainer.train(input=','.join(fins), model_prefix=fmod, vocab_size=vocab_size, character_coverage=character_coverage,	input_sentence_size=input_sentence_size, shuffle_input_sentence=shuffle_input_sentence,	max_sentence_length=max_sentence_length, pad_id=0, pad_piece='<pad>', unk_id=1, unk_piece='<unk>', bos_id=2, bos_piece='<bos>', eos_id=3, eos_piece='<eos>')

	def encode(self, fin, in_type, out_type):
		if in_type == 'int':
			logging.error('SentencePiece tokenizer encoder considers only str as in_type')
			sys.exit()
		out_type = int if out_type == 'int' else str
		raw_lines = fd2list(fin, type=in_type)
		raw_lines = [' '.join(l) for l in raw_lines] ### list of list of words into list of sentences
		tok_lines = self.sp.encode(raw_lines, out_type=out_type)
		return raw_lines, tok_lines

	def decode(self, fin, in_type, out_type):
		if out_type == 'int':
			logging.error('SentencePiece tokenizer decoder considers only str as out_type')
			sys.exit()
		tok_lines = fd2list(fin, type=in_type) #list of list of strings_or_ints		
		raw_lines = self.sp.decode(tok_lines)
		return tok_lines, raw_lines

	def decode_list(self, tok_lines):
		raw_line = self.sp.decode(tok_lines) #raw_line is string		
		return raw_line.split() #list of strings

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

