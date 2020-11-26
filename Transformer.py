# -*- coding: utf-8 -*-

#import io
#import os
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

    toc = time.time()
    logging.info('Done ({:.3f} seconds)'.format(toc-tic))











    
