# Minimalistic NMT toolkit using Transformers

A toolkit developed for research purposes, aiming to be a clean and minimalistic code base achieving similar accuracy than other state-of-the art frameworks.

Built on PyTorch (https://pytorch.org) employing:
* TensorBoard visualizations
* OpenNMT tokenizer (https://github.com/OpenNMT/Tokenizer)
* Google SentencePiece (https://github.com/google/sentencepiece)

## Clients

Vocabulary:
* `tools/build_vocab.py` : Reads (tokenized) training data and outputs a frequency-based vocabulary.

Network:
* `minmt-setup.py` : Creates the NMT network experiment
* `minmt-train.py` : Runs learning 
* `minmt-average.py` : Averages checkpoints
* `minmt-translate.py`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we consider `train.en, train.fr`, `valid.en, valid.fr` and `test.en` the respective train/valid/test files of our running example.
All files are formated with one sentence per line of already tokenized text. You can preprocess your data using any tokenization/sub-tokenization toolkit.
In our `tools` directory we provide several scripts employing libraries of widely used toolkits (BPE/SentencePiece).

### (1) Vocabulary

* Create the vocabulary considered by the network, using:
```
cat train.en | python3 tools/build_vocabulary.py > vocab.en
cat train.fr | python3 tools/build_vocabulary.py > vocab.fr
```
You can use a joint vocabulary:
```
cat train.{en,fr} | python3 tools/build_vocabulary.py > vocab.en-fr
```
By default, the script outputs the 30,000 most frequent tokens appearing in the input training files.

### (2) Create the network

If you built a single vocabulary:
```
python3 minmt-setup.py -dnet $DNET -src_voc vocab.en -tgt_voc vocab.fr
```
Use `vocab.en-fr` for both options if you built a joint vocabulary file.

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
python3 minmt-train.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr
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
-loss NLL
```
```
-shard_size 1000000
-max_length 100
-batch_size 4096
-batch_type tokens
```

Network checkpoints are built in `$DNET` directory named `network.checkpoint_????????.pt`.

### (4) Average checkpoints

Checkpoints available in `$DNET` are averaged in `network.checkpoint_XXXXXXXX_average.pt` with XXXXXXXX the last learning step found.
```
python3 minmt-average -dnet $DNET
```

### (5) Inference

To translate an input file, run the command:
```
python3 minmt-translate.py -dnet $DNET -i test.en
```
The last network checkpoint is considered unless the `-m FILE` option be used.

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

It is highly recommended to use a GPU for training/inference. If you have one, you can prefix the training/inference commands with the `CUDA_VISIBLE_DEVICES=i` envoronment variable as well as adding the `-cuda` option. Ex:

```
CUDA_VISIBLE_DEVICES=0 python3 minmt-train.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr -cuda
CUDA_VISIBLE_DEVICES=0 python3 minmt-translate.py -dnet $DNET -i test.en -cuda
```


