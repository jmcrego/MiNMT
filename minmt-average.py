#!/usr/bin/env python3

import time
import sys
import torch
import logging
import glob
from tools.Tools import create_logger

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

      else:
        self.usage('Unrecognized {} option'.format(tok))

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

  model_files = sorted(glob.glob("{}.checkpoint_????????.pt".format(o.dnet + '/network'))) ### I check if there is one model
  if len(model_files) == 0:
    logging.error('No checkpoint found')
    sys.exit()

  avg_model = None
  final_step = 0
  for i, model_file in enumerate(model_files):
    m = torch.load(model_file, map_location='cpu')
    model = m['model']
    step = m['step']
    logging.info('Loading checkpoint step={} file={}'.format(step,model_file))
    if step > final_step:
      final_step = step 
    if i == 0:
      avg_model = model
    else:
      for (k, v) in avg_model.items():
        avg_model[k].mul_(i).add_(model[k]).div_(i + 1)
  #dump averaged network
  final = {"model": avg_model, "step": final_step}
  torch.save(final, "{}.checkpoint_{:08d}_average.pt".format(o.dnet+'/network',final_step))

  toc = time.time()
  logging.info('Done ({:.2f} seconds)'.format(toc-tic))
