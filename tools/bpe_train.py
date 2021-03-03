# -*- coding: utf-8 -*-
import sys
import os
from Tools import create_logger
from ONMTtok import ONMTtok
import logging
import pyonmttok

if __name__ == '__main__':

  vocab_size = 30000
  model = None
  tok_config = None
  fins = []
  prog = sys.argv.pop(0)
  usage = '''usage: {} -m FILE -i FILE+ [-vocab_size INT]
   -i          FILE : input file/s (multiple files and wildcards allowed)
   -m          FILE : output BPE model
   -vocab_size  INT : vocabulary size ({})
   -tok_config FILE : onmt base tokenization config 
   -h               : this help

Example of tokenization config:
mode: aggressive
segment_case: True
segment_numbers: True
joiner_annotate: True
'''.format(prog,vocab_size)

  while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=='-h':
      sys.stderr.write(usage)
      sys.exit()
    elif tok=='-i' and len(sys.argv)>=0:
      while len(sys.argv)>0 and not sys.argv[0].startswith('-'):
        fins.append(sys.argv.pop(0))
    elif tok=='-m' and len(sys.argv)>=0:
      model = sys.argv.pop(0)
    elif tok=='-vocab_size' and len(sys.argv)>=0:
      vocab_size = int(sys.argv.pop(0))
    elif tok=='-tok_config' and len(sys.argv)>=0:
      tok_config = sys.argv.pop(0)
    else:
      sys.stderr.write('error: unparsed {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  if len(fins)==0:
    sys.stderr.write('error: missing -i option\n')
    sys.exit()

  if model is None:
    sys.stderr.write('error: missing -o option\n')
    sys.exit()

  if tok_config is None:
    sys.stderr.write('error: missing -tok_config option\n')
    sys.exit()

  create_logger('stderr','info')
  tokenizer = ONMTtok(tok_config) #pyonmttok.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True)
  #learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=vocab_size)

  for f in fins:
    logging.info('Ingest: {}'.format(f))
    #learner.ingest_file(f)
  logging.info('Learning bpe model: {}\n'.format(model))
  #tokenizer = learner.learn(model)
