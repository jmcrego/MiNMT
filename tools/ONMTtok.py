# -*- coding: utf-8 -*-
import logging
import yaml
import sys
import os
import pyonmttok

class ONMTtok():

    def __init__(self, fyaml, bpe_model=None):
        opts = {}
        if fyaml is None:
            self.tokenizer = None
        else:
            if not os.path.exists(fyaml):
                logging.error('missing {} file'.format(fyaml))
                sys.exit()
            with open(fyaml) as yamlfile: 
                opts = yaml.load(yamlfile, Loader=yaml.FullLoader)

            if 'mode' not in opts:
                logging.error('Missing mode in tokenizer')
                sys.exit()

            mode = opts["mode"]
            del opts["mode"]
            if bpe_model is not None:
                opts['bpe_path_file'] = bpe_model
            self.tokenizer = pyonmttok.Tokenizer(mode, **opts)
            logging.debug('Built tokenizer mode={} {}'.format(mode,opts))

    def tokenize(self, text):
        if self.tokenizer is None:
            tokens = text.split()
        else:
            tokens, _ = self.tokenizer.tokenize(text)
        return tokens

    def detokenize(self, tokens):
        if self.tokenizer is None:
            return tokens
        return self.tokenizer.detokenize(tokens)
