# -*- coding: utf-8 -*-
import sys
import os
import glob
from SentencePiece import SentencePiece
from Tools import create_logger

if __name__ == '__main__':

  fmod = None
  fin = None
  prog = sys.argv.pop(0)
  usage = '''usage: {} -model FILE
   -model      FILE : input model/vocab
   -i          FILE : input file (stdin)
   -h               : this help
'''.format(prog)

  while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=='-h':
      sys.stderr.write(usage)
      sys.exit()
    elif tok=='-i' and len(sys.argv)>=0:
      fin = sys.argv.pop(0)
    elif tok=='-model' and len(sys.argv)>=0:
      fmod = sys.argv.pop(0)
    else:
      sys.stderr.write('error: unparsed {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  if fmod is None or not os.path.exists(fmod):
    sys.stderr.write('error: missing or unreachable -model file\n')
    sys.exit()

  create_logger('stderr','info')
  mod = SentencePiece(fmod=fmod)

  _, lines = mod.decode(fin=fin, in_type='str', out_type='str')
  for l in lines:
    print(' '.join(l))


