# -*- coding: utf-8 -*-

import sys
import time
import logging
from Params import Params

######################################################################
### MAIN #############################################################
######################################################################
            
if __name__ == '__main__':

    pars = Params(sys.argv)
    tic = time.time()

    if pars.run == 'learning':
      logging.info('Running: learning')

    elif pars.run == 'inference':
      logging.info('Running: inference')

    else:
      logging.warning('Nothing to run')

    toc = time.time()
    logging.info('Done ({:.3f} seconds)'.format(toc-tic))











    
