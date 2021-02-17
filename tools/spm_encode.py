# -*- coding: utf-8 -*-
import sys
import os
import glob
from Preprocessor import SentencePiece
from Tools import create_logger

if __name__ == '__main__':

  fmod = None
  fin = None
  out_type = 'str'
  prog = sys.argv.pop(0)
  usage = '''usage: {} -sp_model FILE [-i FILE]
   -sp_model  FILE : input model/vocab prefix
   -i         FILE : input file (default stdin)
   -out_type  TYPE : str OR int (default str)
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
    elif tok=='-out_type' and len(sys.argv)>=0:
      out_type = sys.argv.pop(0)
    else:
      sys.stderr.write('error: unparsed {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  if sp_model is None or not os.path.exists(sp_model):
    sys.stderr.write('error: missing or unreachable -sp_model file\n')
    sys.exit()

  create_logger('stderr','info')
  sp = SentencePiece(fmod=sp_model)
  _, lines = sp.encode(fin=fin, out_type=str if out_type=='str' else int)
  for l in lines:
    if out_type == 'str':
      print(' '.join(l))
    else:
      print(' '.join([str(x) for x in l]))

