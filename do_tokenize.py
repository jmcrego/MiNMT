# -*- coding: utf-8 -*-
import pyonmttok
import sys

prog = sys.argv.pop(0)
if len(sys.argv) == 0:
    sys.stderr.write('usage: {} bpe_model < raw > tokenized\n'.format(prog))
    sys.stderr.write('Tokenizes raw text using bpe_model\n'.format(prog))
    sys.exit()

bpe_model = sys.argv.pop(0)
tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True, bpe_model_path=bpe_model)

for l in sys.stdin:
    tokens, _ = tokenizer.tokenize(l)
    print(' '.join(tokens))
sys.stderr.write('Done\n')
