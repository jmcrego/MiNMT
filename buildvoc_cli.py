# -*- coding: utf-8 -*-

import sys
import time
import logging
from Vocab import Vocab
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

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

    prog = sys.argv.pop(0)
    usage = '''usage: {} [-tokenizer_config FILE] [-min_freq N] [-max_size N] < text > vocab
   -tokenizer_config FILE : tokenizer config file (if not used mode=space)
   -min_freq          INT : minimum frequence to keep a word (default 1)
   -max_size          INT : maximum number of words in vocab (default 0:all) 
   -log_level       LEVEL : log level [debug, info, warning, critical, error] (info)

further details on onmt-tokenizer at: https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python
'''.format(prog)

    ftokconf = None
    min_freq = 1
    max_size = 0
    log_level = 'info'
    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write(usage);
            sys.exit()
        elif tok=="-tokenizer_config":
            ftokconf = sys.argv.pop(0)
        elif tok=="-min_freq":
            min_freq = int(sys.argv.pop(0))
        elif tok=="-max_size":
            max_size = int(sys.argv.pop(0))
        elif tok=="-log_level":
            log_level = sys.argv.pop(0)

    create_logger(None, log_level)
    logging.info('min_freq={}'.format(min_freq))
    logging.info('max_size={}'.format(max_size))
    token = ONMTTokenizer()
    if ftokconf is not None:
        token.update_yaml(ftokconf)
    token.build()
    tic = time.time()
    voc = Vocab(token)
    voc.build(min_freq,max_size)
    voc.dump()
    toc = time.time()
    logging.info('Done ({:.3f} seconds)'.format(toc-tic))
