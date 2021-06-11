#!/usr/bin/env python3

import sys
import logging
from tools.Tools import create_logger
from collections import Counter

if __name__ == '__main__':

  max_size = 30000
  min_freq = 1
  max_eos = 0
  prog = sys.argv.pop(0)
  usage = '''usage: {} [-min_freq N] [-max_size N] [-max_eos N] < text > vocab
   -min_freq INT : minimum frequence to keep a word, 1 keeps all (default {})
   -max_size INT : maximum number of words in vocab, 0 keeps all (default {})
   -max_eos  INT : adds <eos:N> tokens if N>0                    (default {})
   -h            : this help
Tokens always used:
<pad>
<unk>
<bos>
<eos>
<eos:N>
'''.format(prog,min_freq,max_size,max_eos)

  while len(sys.argv):
    tok = sys.argv.pop(0)
    if tok=="-h":
      sys.stderr.write(usage);
      sys.exit()
    elif tok=="-min_freq":
      min_freq = int(sys.argv.pop(0))
    elif tok=="-max_size":
      max_size = int(sys.argv.pop(0))
    elif tok=="-max_eos":
      max_eos = int(sys.argv.pop(0))

    else:
      sys.stderr.write('Unrecognized {} option\n'.format(tok))
      sys.stderr.write(usage)
      sys.exit()

  create_logger(None, 'info')
  logging.info('min_freq = {}'.format(min_freq))
  logging.info('max_size = {}'.format(max_size))
  logging.info('max_eos = {}'.format(max_eos))

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
  n = 4
  f = 0
  for i in range(max_eos):
    print('<eos:{}>'.format(i))
    n += 1

  for tok, count in freq.most_common():
    if max_size and n >= max_size: ### already output max_size tokens
      break
    if count < min_freq: ### lower frequency than threshold
      break
    if tok=='<pad>' or tok=='<unk>' or tok=='<bos>' or tok=='<eos>': ### reserved token
      continue
    if max_eos>0 and tok.startswith('<eos:') and tok.endswith('>') and tok[5:-1].isdigit(): ### reserved token
      continue
    print(tok)
    f = count
    n += 1
  logging.info('Dumped vocab with {} entries (lowest frequence is {})'.format(n,f))


