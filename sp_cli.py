# -*- coding: utf-8 -*-
import pyonmttok
import os
import sys
import yaml
import logging

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

vocab_size = 30000
character_coverage=0.98
sp_model = None
log_level = 'info'
prog = sys.argv.pop(0)
usage = '''usage: {} -sp_model FILE [-vocab_size INT] [-character_coverage FLOAT]
	-sp_model            FILE : output model file
	-vocab_size           INT : vocabulary size (default {})
	-character_coverage FLOAT : coverage (default {})
'''.format(prog,vocab_size,character_coverage)

while len(sys.argv):
	tok = sys.argv.pop(0)
	if tok=='-h':
		sys.stderr.write(usage)
		sys.exit()
	elif tok=='-sp_model' and len(sys.argv)>=0:
		sp_model = sys.argv.pop(0)
	elif tok=='-vocab_size' and len(sys.argv)>=0:
		vocab_size = int(sys.argv.pop(0))
	elif tok=='-character_coverage' and len(sys.argv)>=0:
		character_coverage = float(sys.argv.pop(0))
	elif tok=="-log_level":
		log_level = sys.argv.pop(0)
	else:
		sys.stderr.write('error: unparsed {} option\n'.format(tok))
		sys.stderr.write(usage)
		sys.exit()

create_logger(None, log_level)

if sp_model is None:		
    logging.error('error: missing -sp_model option')
    sys.exit()

sp_model = os.path.abspath(sp_model)

###
### LEARN SP model
###
learner = pyonmttok.SentencePieceLearner(vocab_size=vocab_size, character_coverage=character_coverage, keep_vocab=True)
for l in sys.stdin:
  learner.ingest(l)
logging.info('Learning sp model: {}'.format(sp_model))
tokenizer = learner.learn(sp_model)

logging.info('Built file {}'.format(sp_model))


