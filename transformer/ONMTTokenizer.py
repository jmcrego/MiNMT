# -*- coding: utf-8 -*-

import sys
#import os
import yaml
import pyonmttok
import logging
#import operator
#import pickle
#import numpy as np
#from collections import defaultdict

####################################################################
### ONMTTokenizer ##################################################
####################################################################
class ONMTTokenizer():
  def __init__(self, fyaml=None, opts=None):
    self.opts = {}
    self.mode = 'space'
    if fyaml is not None:
      self.update_yaml(fyaml)
      self.build()
    elif opts is not None:
      self.update_opts(opts)
      self.build()

  def update_yaml(self, fyaml):
    with open(fyaml) as yamlfile: 
      self.opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if 'mode' not in self.opts:
      logging.error('Missing mode in yaml file')
      sys.exit()
    self.mode = self.opts['mode']
    del self.opts["mode"]
    logging.debug('Update yaml mode={} {}'.format(self.mode,self.opts))

  def update_opts(self, opts, mode=None):
    self.opts = opts
    if 'mode' in self.opts:
      self.mode = self.opts['mode']
      del self.opts['mode']
    if mode is not None:
      self.mode = mode
    logging.debug('Update opts mode={} {}'.format(self.mode,self.opts))

  def build(self, opts=None):
    self.tokenizer = pyonmttok.Tokenizer(self.mode, **self.opts)
    logging.debug('Built tokenizer mode={} {}'.format(self.mode,self.opts))

  def tokenize(self, text):
    return self.tokenizer.tokenize(text)[0]

  def detokenize(self, tokens):
    return self.tokenizer.detokenize(tokens)

