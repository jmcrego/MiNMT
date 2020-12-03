# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import torch

##############################################################################################################
### Inference ################################################################################################
##############################################################################################################
class Inference():
  def __init__(self, model, oi): 
    super(Inference, self).__init__()
    self.model = model

  def translate(self, testset):
    logging.info('Running: inference')

    for i_batch, batch in enumerate(testset):
      pass
    return










