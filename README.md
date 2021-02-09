# Minimalistic Transformer implementation for Machine Translation using Pytorch

## Clients

Preprocessing:
* `sentencepiece_cli` : Learns SentencePiece model over raw (untokenized) text files

Network:
* `create_cli` : Creates network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we considier `$TRAIN`, `$VALID` and `$TEST` variables containing suffixes of the respective train/valid/test files, with `$SS` and `$TT` variables indicating file extensions of source and target language sides.
Train/Valid/Test files are formated with one sentence per line of untokenized (raw) text. 

### (1) Preprocessing

* Build tokenization (SentencePiece) model:
```
cat $TRAIN.{$SS,$TT} | python3 sentencepiece_cli.py -sp_model $SP_JOINT
```
A single SentencePiece model `$SP_JOINT.model` is built for both, source and target, sides of parallel data. 
The script also outputs a vocabulary `$SP_JOINT.vocab` containing the 30,000 most frequent words.

You can use separate SentencePiece models/vocabularies for source and target data sides:
```
cat $TRAIN.$SS | python3 sentencepiece_cli.py -sp_model $SP_SRC
cat $TRAIN.$TT | python3 sentencepiece_cli.py -sp_model $SP_TGT
```

Thus obtaining `$SP_SRC.model`, `$SP_SRC.vocab`, `$SP_TGT.model` and `$SP_TGT.vocab` files.

Skip this preprocessing step if your data is already tokenized.


### (2) Create network


If you built a single model/vocabulary:
```
python3 ./create_cli.py -dnet $DNET -src_vocab $SP_JOINT.vocab -tgt_vocab $SP_JOINT.vocab -src_token $SP_JOINT.model -tgt_token $SP_JOINT.model
```

Otherwise:
```
python3 ./create_cli.py -dnet $DNET -src_vocab $SP_SRC.vocab -tgt_vocab $SP_TGT.vocab -src_token $SP_SRC.model -tgt_token $SP_TGT.model
```

If you skipped preprocessing -src_tok and/or -tgt_tok options are not used.

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
```
python3 ./train_cli.py -dnet $DNET -src_train $TRAIN.$SS -tgt_train $TRAIN.$TT -src_valid $VALID.$SS -tgt_valid $VALID.$TT
```

Starts (or continues) learning using the given training/validation files. Default learning options are:
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
```
python3 ./translate_cli.py -dnet $DNET -i $TEST.$SS
```

Translates the given input file using the last network checkpoint available in `$DNET`. Default inference options are:
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

Remember that test handled are indexed using the tokenization and vocabularies available in $DNET directory.



