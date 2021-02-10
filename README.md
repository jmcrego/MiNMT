# Minimalistic Transformer implementation using Pytorch performing Machine Translation 

## Clients

Preprocessing:
* `sentencepiece_cli` : Learns SentencePiece model over raw (untokenized) text files

Network:
* `create_cli` : Creates the NMT network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we considier `train.en, train.fr`, `valid.en, valid.fr` and `test.en, test.fr` files of the respective train/valid/test files.
Files are formated with one sentence per line of untokenized (raw) text. 

### (1) Preprocessing

* Build a tokenization (SentencePiece) model and vocabulary:
```
cat train.{en,fr} | python3 sentencepiece_cli.py -sp_model SP_joint
```
The script outputs `SP_joint.model` and `SP_joint.vocab` files for both, source and target, sides of parallel data. 
By default, the vocabulary contains the 30,000 most frequent words.

You can use separate SentencePiece models/vocabularies for source and target data sides:
```
cat train.en | python3 sentencepiece_cli.py -sp_model SP_en
cat train.fr | python3 sentencepiece_cli.py -sp_model SP_fr
```

Thus obtaining `SP_en.model`, `SP_en.vocab`, `SP_fr.model` and `SP_fr.vocab`.

* Skip the previous step if your data is already tokenized. You only need to build vocabularies:

A single vocabulary for both data sides:
```
cat train.{en,fr} | python3 tools/buildvoc.py > VOC_joint
```
or one vocabulary for each size:
```
cat train.en | python3 tools/buildvoc.py > VOC_en
cat train.fr | python3 tools/buildvoc.py > VOC_fr
```


### (2) Create the network


If you built a single model/vocabulary:
```
python3 ./create_cli.py -dnet $DNET -src_vocab SP_joint.vocab -tgt_vocab SP_joint.vocab -src_token SP_joint.model -tgt_token SP_joint.model
```

Otherwise:
```
python3 ./create_cli.py -dnet $DNET -src_vocab SP_en.vocab -tgt_vocab SP_fr.vocab -src_token SP_en.model -tgt_token SP_fr.model
```

Do not use `-src_tok` and/or `-tgt_tok` options if you skipped preprocessing. Use `-src_vocab` and `-tgt_vocab` with the corresponding vocabularies if you used `tools/buildvoc.py`

The script creates the directory `$DNET` containing:
* A network description file: 
  * network
* Vocabularies and tokenization (SentencePiece) models:
  * src_voc
  * tgt_voc
  * src_tok (if used -src_token)
  * tgt_tok (if used -tgt_token)

Default network options are:
```
-emb_dim 512
-qk_dim 64
-v_dim 64
-ff_dim 2048
-n_heads 8
-n_layers 6
-dropout 0.1
-share_embeddings False
```

### (3) Learning

Use the next command:
```
python3 ./train_cli.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr
```
to start (or continue) learning using the given training/validation files. 

Default learning options are:
```
-max_steps 0
-max_epochs 0
-validate_every 5000
-save_every 5000
-report_every 100
-keep_last_n 10
-clip_grad_norm 0.0
```
```
-lr 2.0
-min_lr 0.0001
-beta1 0.9
-beta2 0.998
-eps 1e-09
-noam_scale 2.0
-noam_warmup 4000
-label_smoothing 0.1
-loss KLDiv
```
```
-shard_size 1000000
-max_length 100
-batch_size 4096
-batch_type tokens
```

Network checkpoints are built in `$DNET` directory named `network.checkpoint_????????.pt`.

Remember that training and validation datasets are handled using the tokenization and vocabularies available in $DNET directory.

### (4) Inference

Use the next command:
```
python3 ./translate_cli.py -dnet $DNET -i test.en
```
to translate the given input file using the last network checkpoint available in `$DNET`. 

Default inference options are:
```
-beam_size 4
-n_best 1
-max_size 250
-alpha 0.0
-format iH
```
```
-shard_size 0
-max_length 0
-batch_size 30
-batch_type sentences
```

Option -format specifies the fields to output for every example (TAB-separated):
```
[i] index in test set (sentences are sorted to minimize padding)
[n] rank in n-best
[c] global hypothesis cost
[s] source sentence
[S] source sentence (detokenised)
[u] source indexes
[h] hypothesis
[H] hypothesis (detokenised)
[v] hypothesis indexes
```

Remember that test datasets are handled using the tokenization and vocabularies available in $DNET directory.



