# -*- coding: utf-8 -*-
import pyonmttok
import sys
import yaml

symbols = 30000
tok_config = None
bpe_model = None
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
	else:
		sys.stderr.write('error: unparsed {} option\n'.format(tok))
		sys.stderr.write(usage)
		sys.exit()

if tok_config is None:		
	sys.stderr.write('error: missing -tok_config option\n')
	sys.stderr.write(usage)
	sys.exit()

if bpe_model is None:		
	sys.stderr.write('error: missing -bpe_model option\n')
	sys.stderr.write(usage)
	sys.exit()

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
sys.stderr.write('Using tokenization mode: {} opts: {}\n'.format(mode, opts))

###
### LEARN BPE model
###
tokenizer = pyonmttok.Tokenizer(mode, **opts)
learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=symbols)
for l in sys.stdin:
  learner.ingest(l)
sys.stderr.write('learning bpe model: {}\n'.format(bpe_model))
tokenizer = learner.learn(bpe_model)

###
### CONFIG FILE
###
opts['mode'] = mode
opts['bpe_model_path'] = bpe_model
with open('{}.tok_config'.format(bpe_model), "w") as yamlfile:
	yaml.dump(opts, yamlfile)

sys.stderr.write('Built {} and {}.tok_config files\n'.format(bpe_model,bpe_model))
