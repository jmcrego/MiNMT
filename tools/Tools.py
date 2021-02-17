# -*- coding: utf-8 -*-

import sys
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


def isbinary(fin):
  try:
    with open(fin, "r") as f:
      n = 0
      for l in f:
        n += 1
        if n == 10: 
          break
      return False
  except UnicodeDecodeError: # Fond non-text data  
    return True
