# -*- coding: utf-8 -*-
import sys
import logging
from ONMTTokenizer import ONMTTokenizer

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
log_level = 'info'
prog = sys.argv.pop(0)
usage = '''usage: {} -sp_model FILE [-log_level LEVEL] < stdin > stdout
   -sp_model   FILE : SentencePiece model file (space tokenizer)
   -log_level LEVEL : log level [debug, info, warning, critical, error] ({})
   -h               : this help
'''.format(prog,log_level)

while len(sys.argv):
  tok = sys.argv.pop(0)
  if tok=='-h':
    sys.stderr.write(usage)
    sys.exit()
  elif tok=='-sp_model' and len(sys.argv)>=0:
    sp_model = sys.argv.pop(0)
  elif tok=="-log_level":
    log_level = sys.argv.pop(0)
  else:
    sys.stderr.write('error: unparsed {} option\n'.format(tok))
    sys.stderr.write(usage)
    sys.exit()

create_logger(None, log_level)

token = ONMTTokenizer(sp_model) ### sp_model may be None => space tokenizer
for l in sys.stdin:
  t = token.tokenize(l.rstrip())
  print(' '.join(t))
