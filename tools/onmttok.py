# -*- coding: utf-8 -*-
import yaml
import sys
import os
import pyonmttok

class onmttok():

    def __init__(self, fyaml, bpe_model=None, sp_model=None):
        opts = {}
        if fyaml is None:
            opts['mode'] = 'space' ### this is used for sp_model
        elif not os.path.exists(fyaml):
            sys.stderr.write('Error: cannot find fyaml={}\n'.format(fyaml))
            sys.exit()
        else:
            with open(fyaml) as yamlfile: 
                opts = yaml.load(yamlfile, Loader=yaml.FullLoader)

        if 'mode' not in opts:
            sys.stderr.write('Error: missing mode in tokenizer config\n')
            sys.exit()

        mode = opts["mode"]
        del opts["mode"]

        if bpe_model is not None:
            opts['bpe_model_path'] = bpe_model
        elif sp_model is not None:
            opts['sp_model_path'] = sp_model
        self.tokenizer = pyonmttok.Tokenizer(mode, **opts)
        sys.stderr.write('Built tokenizer mode={} {}\n'.format(mode,opts))

    def tokenize(self, text):
        tokens, _ = self.tokenizer.tokenize(text)
        return tokens

    def detokenize(self, tokens):
        return self.tokenizer.detokenize(tokens)


def learn_bpe(tok_config, bpe_model, symbols=32000, files=[]):
    tokenizer = onmttok(tok_config)
    learner = pyonmttok.BPELearner(tokenizer=tokenizer.tokenizer, symbols=symbols)

    if len(files):
        for f in files:
            sys.stderr.write('Ingest file={}\n'.format(f))
            sys.stderr.flush()
            learner.ingest_file(f)
    else:
        sys.stderr.write('Ingest stdin\n')
        sys.stderr.flush() 
        for l in sys.stdin:
            learner.ingest(l)
    sys.stderr.write('Learning {}\n'.format(bpe_model))
    sys.stderr.flush()
    learner.learn(bpe_model)

def learn_sp(sp_model, vocab_size=32000, character_coverage=0.98, files=[]):
    learner = pyonmttok.SentencePieceLearner(vocab_size=vocab_size, character_coverage=character_coverage)

    if len(files):
        for f in files:
            sys.stderr.write('Ingest file={}\n'.format(f))
            sys.stderr.flush()
            learner.ingest_file(f)
    else:
        sys.stderr.write('Ingest stdin\n') 
        sys.stderr.flush()
        for l in sys.stdin:
            learner.ingest(l)
    sys.stderr.write('Learning {}\n'.format(sp_model))
    sys.stderr.flush()
    learner.learn(sp_model)    
