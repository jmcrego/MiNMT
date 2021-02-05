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

symbols = 30000
tok_config = None
bpe_model = None
log_level = 'info'
prog = sys.argv.pop(0)
usage = '''usage: {} -tok_config FILE -bpe_model FILE [-symobls INT]
	-tok_config FILE : tokenization config file
	-bpe_model  FILE : output model file
	-symbols     INT : number of operations (default 30000)
'''.format(prog,symbols)

while len(sys.argv):
	tok = sys.argv.pop(0)
	if tok=='-h':
		sys.stderr.write(usage)
		sys.exit()
	elif tok=='-tok_config' and len(sys.argv)>=0:
		tok_config = sys.argv.pop(0)
	elif tok=='-bpe_model' and len(sys.argv)>=0:
		bpe_model = sys.argv.pop(0)
	elif tok=='-symbols' and len(sys.argv)>=0:
		symbols = int(sys.argv.pop(0))
	elif tok=="-log_level":
		log_level = sys.argv.pop(0)
	else:
		sys.stderr.write('error: unparsed {} option\n'.format(tok))
		sys.stderr.write(usage)
		sys.exit()

create_logger(None, log_level)

if tok_config is None:		
	logging.error('error: missing -tok_config option')
	sys.exit()

if bpe_model is None:		
	logging.error('error: missing -bpe_model option')
	sys.exit()
bpe_model = os.path.abspath(bpe_model)

###
### READ tokenization config file
###
with open(tok_config) as yamlfile: 
	opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
if 'mode' not in opts:
	logging.error('Missing mode in yaml file')
	sys.exit()
mode = opts['mode']
del opts["mode"]
logging.info('Tokenization mode: {} opts: {}'.format(mode, opts))

###
### LEARN BPE model
###
tokenizer = pyonmttok.Tokenizer(mode, **opts)
learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=symbols)
for l in sys.stdin:
  learner.ingest(l)
logging.info('Learning bpe model: {}'.format(bpe_model))
tokenizer = learner.learn(bpe_model)

###
### OUTPUT config file with bpe model
###
opts['mode'] = mode
opts['bpe_model_path'] = bpe_model
with open('{}.tok_config'.format(bpe_model), "w") as yamlfile:
	yaml.dump(opts, yamlfile)

logging.info('Built {} and {}.tok_config files'.format(bpe_model,bpe_model))


