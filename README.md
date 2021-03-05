# Minimalistic NMT toolkit using Transformers

A toolkit built on PyTorch (https://pytorch.org) developed for research purposes, aiming to be a clean and minimalistic code base achieving similar accuracy than other state-of-the art frameworks.

## Clients

* `minmt-vocab.py` : Reads (tokenized) training data and outputs a frequency-based vocabulary
* `minmt-setup.py` : Creates the NMT network experiment
* `minmt-train.py` : Runs learning 
* `minmt-average.py` : Averages checkpoints
* `minmt-translate.py`: Runs inference


:information_source: Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we consider `train.en, train.fr`, `valid.en, valid.fr` and `test.en` the respective train/valid/test files of our running example.
All files are formated one sentence per line of already preprocessed (tokenized) text. 

You must preprocess your data using any tokenization/sub-tokenization toolkit.
We provide several scripts making use of the OpenNMT tokenizer library (https://github.com/OpenNMT/Tokenizer) implementing widely used tokenization/sub-tokenization algorithms (BPE/SentencePiece).
* `tools/learn_bpe.py` : learns BPE model
* `tools/learn_sp.py` : learns SentencePiece model
* `tools/tokenizer.py` : tokenizes/detokenizes using the previous sub-tokenization models

### (1) Vocabulary

* Create the (source/target) vocabularies considered by the network:
```
$ cat train.en | minmt-vocab.py > vocab.en
$ cat train.fr | minmt-vocab.py > vocab.fr
```
If you prefere, you can use a single joint vocabulary for both languages:
```
$ cat train.{en,fr} | minmt-vocab.py > vocab.en-fr
```
Default vocabulary options are:
```
-max_size 30000
-min_freq 1
```

### (2) Create the network

To setup the network follow:
```
minmt-setup.py -dnet $DNET -src_voc vocab.en -tgt_voc vocab.fr
```
Use `vocab.en-fr` in both (-src_voc and -tgt_voc) options if you built a joint vocabulary file.

The script creates the directory `$DNET` containing:
* `network` (the network description file)
* `src_voc` (source-side vocabulary)
* `tgt_voc` (target-side vocabulary)

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

To start (or continue) learning, run the command:
```
$ minmt-train.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr
```

Default learning options are:
```
-max_steps 0
-max_epochs 0
-validate_every 5000
-save_every 5000
-report_every 100
-keep_last_n 10
-clip_grad_norm 0.0

-lr 2.0
-min_lr 0.0001
-beta1 0.9
-beta2 0.998
-eps 1e-09
-noam_scale 2.0
-noam_warmup 4000
-label_smoothing 0.1
-loss NLL

-shard_size 1000000
-max_length 100
-batch_size 4096
-batch_type tokens
```

Network checkpoints are built in `$DNET` directory named `network.checkpoint_????????.pt` with `????????` being the learning step.

### (4) Average checkpoints

Checkpoints available in `$DNET` can be averaged running:
```
$ minmt-average -dnet $DNET
```
The resulting network is available in `network.checkpoint_????????_average.pt`. Averaging last checkpoints typically results in a light performance improvement.


### (5) Inference

To translate an input file, run the command:
```
$ minmt-translate.py -dnet $DNET -i test.en
```
The last network checkpoint is considered unless the `-m FILE` option be used.

Default inference options are:
```
-beam_size 4
-n_best 1
-max_size 250
-alpha 0.0
-format pt

-shard_size 0
-max_length 0
-batch_size 30
-batch_type sentences
```

Option `-format` specifies the fields to output for every sentence (TAB-separated):
```
[p] position of sentence in test set
[n] rank in n-best
[c] global hypothesis cost
[j] input sentence (ids) : 104 17 71 406 4
[s] input sentence (tok) : ▁This ▁is ▁an ▁example .
[i] hypothesis (ids) : 1738 40 44 551 4
[t] hypothesis (tok) : ▁Ceci ▁est ▁un ▁exemple .
```

## Use of GPU:

It is highly recommended to use a GPU for learning/inference steps. 
If you have one, you can prefix the training/inference commands with the `CUDA_VISIBLE_DEVICES=i` envoronment variable as well as with the `-cuda` option. Ex:

```
$ CUDA_VISIBLE_DEVICES=0 minmt-train.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr -cuda
$ CUDA_VISIBLE_DEVICES=0 minmt-translate.py -dnet $DNET -i test.en -cuda
```


