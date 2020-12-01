# -*- coding: utf-8 -*-

import sys
import os
import logging
import pickle
import numpy as np
import torch
from collections import defaultdict

##############################################################################################################
### Learning #################################################################################################
##############################################################################################################
class Learning():
  def __init__(self, o, model, optim): 
    super(Learning, self).__init__()
    self.model = model
    self.optim = optim
    self.max_steps = o.max_steps
    self.max_epochs = o.max_epochs
    self.validate_every = o.validate_every
    self.save_every = o.save_every
    self.report_every = o.report_every
    self.keep_last_n = o.keep_last_n
    self.step = 0
    self.epoch = 0

  def learn(self, trainset, validset):
    while True:
      self.epoch += 1
      logging.info('Epoch {}'.format(self.epoch))

      trainset.shuffle()
      for batch in trainset:
        self.step += 1
        self.model.train()
        loss = self.model.forward(batch[0],batch[1],batch[2])
#        self.optim.zero_grad()
#        loss.backward()
#        self.optim.step()

        if self.report_every and self.step % self.report_every == 0:
          logging.info('Learning step {}'.format(self.step))

        if self.validate_every and self.step % self.validate_every == 0:
          vloss = self.validate(validset)

        if self.save_every and self.step % self.save_every == 0:
          save_model(self.name, self.model, self.step, self.keep_last_n)
          save_optim(self.name, self.optim)

        if self.max_steps and self.step >= self.max_steps:
          vloss = self.validate(validset)
          save_model(self.name, self.model, self.step, self.keep_last_n)
          save_optim(self.name, self.optim)
          return

      if self.max_epochs and self.epoch >= self.max_epochs:
        vloss = self.validate(validset)
        save_model(self.name, self.model, self.step, self.keep_last_n)
        save_optim(self.name, self.optim)
        return
    return

  def validate(self, validset):
    logging.info('Validation step {}'.format(self.step))
#    with torch.no_grad():
#      model.eval()
    return 0.0









