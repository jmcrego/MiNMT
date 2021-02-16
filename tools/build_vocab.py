# -*- coding: utf-8 -*-

import sys
import logging
from Tools import create_logger
from collections import defaultdict


if __name__ == '__main__':

  max_size = 30000
  min_freq = 1
  prog = sys.argv.pop(0)
  usage = '''usage: {} [-min_freq N] [-max_size N] < text > vocab
   -min_freq INT : minimum frequence to keep a word, 1 keeps all (default {})
   -max_size INT : maximum number of words in vocab, 0 keeps all (default {})
   -h            : this help
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

  create_logger(None, 'info')
  logging.info('min_freq = {}'.format(min_freq))
  logging.info('max_size = {}'.format(max_size))

  tok_to_frq = defaultdict(int)
  for l in sys.stdin:
    for tok in l.rstrip().split():
      tok_to_frq[tok] += 1
  logging.info('Read stdin with {} distinct tokens'.format(len(tok_to_frq)))

  seen = defaultdict(int)
  print('<pad>')
  print('<unk>')
  print('<bos>')
  print('<eos>')
  seen['<pad>'] += 1
  seen['<unk>'] += 1
  seen['<bos>'] += 1
  seen['<eos>'] += 1
  for tok, frq in sorted(tok_to_frq.items(), key=lambda item: item[1], reverse=True):
    if (max_size and len(seen) == max_size) or frq < min_freq:
      break
    if tok in seen: #in case reserved words appear in text
      continue
    print(tok)
    seen[tok] += 1
  logging.info('Dumped vocab with {} entries'.format(len(seen)))