#!/usr/bin/env python3

import sys
import logging
from tools.Tools import create_logger
from collections import Counter

if __name__ == '__main__':

  max_size = 30000
  min_freq = 1
  prog = sys.argv.pop(0)
  usage = '''usage: {} [-min_freq N] [-max_size N] < text > vocab
   -min_freq INT : minimum frequence to keep a word, 1 keeps all (default {})
   -max_size INT : maximum number of words in vocab, 0 keeps all (default {})
   -h            : this help
Tokens always used:
<pad>
<unk>
<bos>
<eos>
⸨sep⸩
⸨msk⸩
'''.format(prog,min_freq,max_size)

  while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=="-h":
      sys.stderr.write(usage);
      sys.exit()
    elif tok=="-min_freq":
      min_freq = int(sys.argv.pop(0))
    elif tok=="-max_size":
      max_size = int(sys.argv.pop(0))

    else:
      sys.stderr.write('Unrecognized {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  create_logger(None, 'info')
  logging.info('min_freq = {}'.format(min_freq))
  logging.info('max_size = {}'.format(max_size))

  ###################
  ### count words ###
  ###################
  lflat = []
  ll = [l.split() for l in sys.stdin.readlines()]
  list(map(lflat.extend, ll))
  freq = Counter(lflat)

  #######################
  ### dump vocabulary ###
  #######################
  print('<pad>')
  print('<unk>')
  print('<bos>')
  print('<eos>')
  print('⸨sep⸩')
  print('⸨msk⸩')
  n = 6
  for tok, count in freq.most_common():
    if max_size and n >= max_size:
      break
    if count < min_freq:
      break
    if tok=='<pad>' or tok=='<unk>' or tok=='<bos>' or tok=='<eos>' or tok=='⸨msk⸩' or tok=='⸨sep⸩':
      continue
    print(tok)
    f = count
    n += 1
  logging.info('Dumped vocab with {} entries (lowest frequence is {})'.format(n,f))


