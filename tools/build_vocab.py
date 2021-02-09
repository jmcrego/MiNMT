# -*- coding: utf-8 -*-

import sys
import logging
from ONMTTokenizer import ONMTTokenizer
from collections import defaultdict

def create_logger(logfile, loglevel):
  numeric_level = getattr(logging, loglevel.upper(), None)
  if not isinstance(numeric_level, int):
    logging.error("Invalid log level={}".format(loglevel))
    sys.exit()
  if logfile is None or logfile == 'stderr':
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
    logging.debug('Created Logger level={}'.format(loglevel))
  else:
    logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
    logging.debug('Created Logger level={} file={}'.format(loglevel, logfile))

sp_model = None
vocab_size = None
min_freq = None
log_level = 'info'
prog = sys.argv.pop(0)
usage = '''usage: {} [-vocab_size INT] [-sp_model FILE] [-log_level LEVEL] < stdin
   -vocab_size  INT : vocabulary size ({})
   -min_freq    INT : minimum token frequency ({})
   -sp_model   FILE : SentencePiece model file
   -log_level LEVEL : log level [debug, info, warning, critical, error] ({})
   -h               : this help
[Visit https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python to modify options]
'''.format(prog,vocab_size,min_freq,log_level)

while len(sys.argv):
  tok = sys.argv.pop(0)
  if tok=='-h':
    sys.stderr.write(usage)
    sys.exit()
  elif tok=='-sp_model' and len(sys.argv)>=0:
    sp_model = sys.argv.pop(0)
  elif tok=='-vocab_size' and len(sys.argv)>=0:
    vocab_size = int(sys.argv.pop(0))
  elif tok=='-min_freq' and len(sys.argv)>=0:
    min_freq = int(sys.argv.pop(0))
  elif tok=="-log_level":
    log_level = sys.argv.pop(0)
  else:
    sys.stderr.write('error: unparsed {} option\n'.format(tok))
    sys.stderr.write(usage)
    sys.exit()

create_logger(None, log_level)

Freq = defaultdict(int)
nlines = 0
nwords = 0
token = ONMTTokenizer(sp_model)
for l in sys.stdin:
  nlines += 1
  t = token.tokenize(l.rstrip())

  nlines += 1
  for word in t:
    nwords += 1
    Freq[word] += 1

logging.info("Read #lines={} #words={} vocab={}".format(nlines, nwords, len(Freq)))

for n, (wrd, frq) in enumerate(sorted(Freq.items(), key=lambda item: item[1], reverse=True)):
  if vocab_size is not None and n >= vocab_size:
    break
  if min_freq is not None and frq < min_freq:
    break
  print("{}\t{}".format(wrd,frq))

logging.info("Vocabulary size = {}".format(n))






