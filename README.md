# Minimalistic implementation of a NMT toolkit using Transformers

* PyTorch framework (https://pytorch.org)
* Google SentencePiece (https://github.com/google/sentencepiece)
* TensorBoard visualizations

## Clients

Preprocessing:
* `tools/spm_train.py` : Learns the SentencePiece model over raw (untokenized) text files

Network:
* `create_cli` : Creates the NMT network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we consider `train.en, train.fr`, `valid.en, valid.fr` and `test.en, test.fr` files of the respective train/valid/test files.
Files are formated with one sentence per line of untokenized (raw) text. 

### (1) Preprocessing

* Build a tokenization (SentencePiece) model and vocabulary:
```
python3 tools/spm_train.py -sp_model SP_joint -i train.{en,fr}
```
The script outputs `SP_joint.model` and `SP_joint.vocab` files for both, source and target, sides of parallel data. 
By default, the vocabulary contains the 30,000 most frequent words. The vocabulary is not further needed, already contained in the model file.

You can use two separate SentencePiece model/vocabulary for source and target data sides:
```
python3 sentencepiece_cli.py -sp_model SP_en -i train.en
python3 sentencepiece_cli.py -sp_model SP_fr -i train.fr
```

Thus producing `SP_en.model`, `SP_en.vocab`, `SP_fr.model` and `SP_fr.vocab`.


### (2) Create the network


If you built a single model/vocabulary:
```
python3 ./create_cli.py -dnet $DNET -src_spm SP_joint.model -tgt_spm SP_joint.model
```

Otherwise:
```
python3 ./create_cli.py -dnet $DNET -src_spm SP_en.model -tgt_spm SP_fr.model
```

The script creates the directory `$DNET` containing:
* A network description file: 
  * network
* Vocabularies and tokenization (SentencePiece) models:
  * src_spm
  * tgt_spm

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

Use the command
```
python3 ./train_cli.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr
```
to start (or continue) learning using the given training/validation files. 

Default learning options are
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

Use the command
```
python3 ./translate_cli.py -dnet $DNET -i test.en
```
to translate the given input file using the last network checkpoint available in `$DNET`. 

Default inference options are
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

Option `-format` specifies the fields to output for every example (TAB-separated):
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

Test datasets are handled using the `src_spm` and `tgt_spm` SentencePiece models existing in `$DNET` directory.



