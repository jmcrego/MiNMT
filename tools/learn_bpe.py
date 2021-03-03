#!/usr/bin/env python3

import sys
import os
import time
from onmttok import learn_bpe

if __name__ == '__main__':

    files = []
    symbols = 32000
    bpe_model = None
    tok_config = None
    prog = sys.argv.pop(0)
    usage = '''usage: {} -tok_config FILE -bpe_model FILE [-symbols INT] [-i FILES]
    -bpe_model  FILE : bpe model to build
    -tok_config FILE : base tokenization config file
    -symbols     INT : number of BPE operations ({})
    -i         FILES : comma-separated list of files ({})
    -h               : this help
'''.format(prog,symbols,'stdin')
    
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write(usage);
            sys.exit()
        elif tok=="-symbols" and len(sys.argv):
            symbols = int(sys.argv.pop(0))
        elif tok=="-tok_config" and len(sys.argv):
            tok_config = sys.argv.pop(0)
        elif tok=="-bpe_model" and len(sys.argv):
            bpe_model = sys.argv.pop(0)
        elif tok=="-i" and len(sys.argv):
            files = sys.argv.pop(0).split(',')
        else:
            sys.stderr.write('Error: unparsed {} option\n'.format(tok))
            sys.stderr.write(usage)
            sys.exit()
            
    if bpe_model is None:
        sys.stderr.write('Error: missing -bpe_model option\n')
        sys.exit()

    if tok_config is None:
        sys.stderr.write('Error: missing -tok_config option\n')
        sys.exit()
        
    tic = time.time()
    learn_bpe(tok_config, bpe_model, symbols=symbols, files=files)
    toc = time.time()
    sys.stderr.write('Done ({:.2f} seconds)\n'.format(toc-tic))
