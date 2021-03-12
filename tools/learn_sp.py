#!/usr/bin/env python3

import sys
import os
import time
from onmttok import learn_sp

if __name__ == '__main__':

    files = []
    vocab_size = 32000
    character_coverage = 0.98
    sp_model = None
    prog = sys.argv.pop(0)
    usage = '''usage: {} -sp_model FILE [-vocab_size INT] [-character_coverage FLOAT] [-i FILES]
    -sp_model            FILE : bpe model to build
    -vocab_size           INT : vocabulary size ({})
    -character_coverage FLOAT : character coverage ({})
    -i                  FILES : comma-separated list of files ({})
    -h                        : this help
'''.format(prog,vocab_size,character_coverage,'stdin')
    
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write(usage);
            sys.exit()
        elif tok=="-vocab_size" and len(sys.argv):
            vocab_size = int(sys.argv.pop(0))
        elif tok=="-character_coverage" and len(sys.argv):
            character_coverage = float(sys.argv.pop(0))
        elif tok=="-sp_model" and len(sys.argv):
            sp_model = sys.argv.pop(0)
        elif tok=="-i" and len(sys.argv):
            files = sys.argv.pop(0).split(',')
        else:
            sys.stderr.write('Error: unparsed {} option\n'.format(tok))
            sys.stderr.write(usage)
            sys.exit()
            
    if sp_model is None:
        sys.stderr.write('Error: missing -sp_model option\n')
        sys.exit()

    tic = time.time()
    learn_sp(sp_model, vocab_size=vocab_size, character_coverage=character_coverage, files=files)
    toc = time.time()
    sys.stderr.write('Done ({:.2f} seconds)\n'.format(toc-tic))
