#!/usr/bin/env python3

import time
import sys
import torch
import logging
from tools.Tools import create_logger
from transformer.Model import average_models

######################################################################
### Options ##########################################################
######################################################################

class Options():
  def __init__(self, argv):
    self.prog = argv.pop(0)
    self.dnet = None

    log_file = 'stderr'
    log_level = 'info'

    while len(argv):
      tok = sys.argv.pop(0)
      if tok=="-h":
        self.usage()
      elif tok=="-dnet" and len(argv):
        self.dnet = argv.pop(0)
      elif tok=="-log_file" and len(argv):
        log_file = argv.pop(0)
      elif tok=="-log_level" and len(argv):
        log_level = argv.pop(0)

    create_logger(log_file,log_level)
    if self.dnet is None:
      logging.error('missing -dnet option')
      self.usage()

  def usage(self):
    sys.stderr.write('''usage: {} -dnet DIR [Options]
   -dnet         DIR : network directory

   -log_file    FILE : log file  (stderr)
   -log_level    STR : log level [debug, info, warning, critical, error] (info)
   -h                : this help
'''.format(self.prog))
    sys.exit()

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

  tic = time.time()
  o = Options(sys.argv)
  average_models(o.dnet + '/network')
  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))
