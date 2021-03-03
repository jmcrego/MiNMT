# -*- coding: utf-8 -*-
import sys
import os
from Tools import create_logger
from SentencePiece import SentencePiece

if __name__ == '__main__':

  vocab_size = 30000
  model = None
  fins = []
  prog = sys.argv.pop(0)
  usage = '''usage: {} -m FILE -i FILE+ [-vocab_size INT]
   -i         FILE : input file/s (multiple files and wildcards allowed)
   -m         FILE : output SP model/vocab prefix
   -vocab_size INT : vocabulary size ({})
   -h              : this help
Visit: https://github.com/google/sentencepiece/blob/master/doc/options.md
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

  create_logger('stderr','info')
  sp = SentencePiece(fmod=None)
  sp.train(fmod=model, fins=fins, vocab_size=vocab_size)
