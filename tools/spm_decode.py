# -*- coding: utf-8 -*-
import sys
import os
import glob
from SentencePiece import SentencePiece
from Tools import create_logger

if __name__ == '__main__':

  sp_model = None
  fin = None
  in_type = 'str'
  prog = sys.argv.pop(0)
  usage = '''usage: {} -sp_model FILE [-i FILE]
   -sp_model  FILE : input model/vocab prefix
   -i         FILE : input file (default stdin)
   -in_type   TYPE : str OR int (default str)
   -h              : this help
Visit: https://github.com/google/sentencepiece/blob/master/python/README.md
'''.format(prog)

  while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=='-h':
      sys.stderr.write(usage)
      sys.exit()
    elif tok=='-i' and len(sys.argv)>=0:
      fin = sys.argv.pop(0)
    elif tok=='-sp_model' and len(sys.argv)>=0:
      sp_model = sys.argv.pop(0)
    elif tok=='-in_type' and len(sys.argv)>=0:
      in_type = sys.argv.pop(0)
    else:
      sys.stderr.write('error: unparsed {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  if sp_model is None or not os.path.exists(sp_model):
    sys.stderr.write('error: missing or unreachable -sp_model file\n')
    sys.exit()

  create_logger('stderr','info')
  sp = SentencePiece(sp_model=sp_model)
  _, lines = sp.decode(fin=fin, in_type=str if in_type=='str' else int)
  for l in lines:
    print(l)
