# -*- coding: utf-8 -*-

import sys
import time
import logging
from Data import Vocab

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':
    prog = sys.argv.pop(0)
    if len(sys.argv) == 0:
        sys.stderr.write('usage: {} ONMT_TOKENIZER_CONFIG < text > vocab\n'.format(prog))
        sys.exit()

    tic = time.time()
    voc = Vocab()
    voc.build(sys.argv[0])
    toc = time.time()
    sys.stderr.write('Done ({:.3f} seconds)\n'.format(toc-tic))











    
