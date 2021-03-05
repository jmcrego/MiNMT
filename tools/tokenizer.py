#!/usr/bin/env python3

import sys
import os
import time
from onmttok import onmttok

if __name__ == '__main__':

    bpe_model = None
    sp_model = None
    tok_config = None
    detok = False
    prog = sys.argv.pop(0)
    usage = '''usage: {} -tok_config FILE [-bpe_model FILE] [-sp_model FILE] [-detok] < stdin
    -tok_config FILE : word tokenization config file
    -bpe_model  FILE : bpe model to use
    -sp_model   FILE : sp model to use
    -detok           : run detokenization
    -h               : this help

tok_config FILE is a yaml file describing onmt tokenization (further details in https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python)
BPE requires a word tokenization (mode: aggressive or conservative) while no word tokenization is typically used for SP (mode: space) 

Consider for instance the next tok_config files:
* for BPE:
mode: aggressive
joiner_annotate: True
* for SP:
mode: space
spacer_annotate: True
'''.format(prog)

    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write(usage);
            sys.exit()
        elif tok=="-tok_config" and len(sys.argv):
            tok_config = sys.argv.pop(0)
        elif tok=="-bpe_model" and len(sys.argv):
            bpe_model = sys.argv.pop(0)
        elif tok=="-sp_model" and len(sys.argv):
            sp_model = sys.argv.pop(0)
        elif tok=="-detok":
            detok = True
        else:
            sys.stderr.write('Error: unparsed {} option\n'.format(tok))
            sys.stderr.write(usage)
            sys.exit()

    if tok_config is None:
        sys.stderr.write('Error: missing -tok_config option\n')
        sys.exit()

    if bpe_model is not None and sp_model is not None:
        sys.stderr.write('Error: only one of -bpe_model and -sp_model can be used\n')
        sys.exit()

    tokenizer = onmttok(tok_config, bpe_model=bpe_model, sp_model=sp_model)
    tic = time.time()
    for l in sys.stdin:
        l = l.rstrip()
        if detok:
            #print('detok:',l.split())
            print(tokenizer.detokenize(l.split()))
        else:
            #print('tok:',tokenizer.tokenize(l))
            print(' '.join(tokenizer.tokenize(l)))
    toc = time.time()
    sys.stderr.write('Done ({:.2f} seconds)\n'.format(toc-tic))

    

