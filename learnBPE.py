# -*- coding: utf-8 -*-
import pyonmttok
import sys

prog = sys.argv.pop(0)
if len(sys.argv) == 0:
    sys.stderr.write('usage: {} bpe_model < raw\n'.format(prog))
    sys.stderr.write('builds bpe_model using raw text\n'.format(prog))
    sys.exit()

bpe_model = sys.argv.pop()

tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True)
learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=32000)

for l in sys.stdin:
    learner.ingest(l)

sys.stderr.write('learning bpe model: {}\n'.format(bpe_model))
tokenizer = learner.learn(bpe_model)
