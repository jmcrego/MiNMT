# Minimalistic Transformer implementation for Machine Translation using Pytorch

## Clients

Preprocess:
* `buildBPE_cli` : Learns BPE model
* `buildVOC_cli` : Builds vocabulary

Network:
* `create_cli` : Creates network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we considier `$TRAIN`, `$VALID` and `$TEST` variables containing suffixes of the respective train/valid/test files, with `$SS` and `$TT` variables indicating file extensions of source and target language sides.
Train/Valid/Test files are formated with one sentence per line of untokenized (raw) text. 

### (1) Preprocess

Preprocessing indicates the string transformations performed over raw text files before passed to the NMT network. Namely, tokenization and subtokenization steps. We use the python api (https://github.com/OpenNMT/Tokenizer) that can be installed via `pip install pyonmttok`.

* Create `$TOK` (tokenization config file) containing tokenization options. For instance:
```
mode: aggressive
joiner_annotate: True
segment_numbers: True
```

If text files are already tokenized you can use the next tokenization options (and skip learning BPE model):
```
mode: space
```

Further information on tokenization options in https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python 


* Build `$BPE` Model:
```
cat $TRAIN.{$SS,$TT} | python3 buildBPE_cli.py -tok_config $TOK -bpe_model $BPE
```
A single BPE model is built for both, source and target, sides of parallel data.
Previous to BPE learning, the input stream is tokenized as detailed in `$TOK`.
Output consists of the BPE model `$BPE` and a new tokenization config file containing a reference to the BPE model `$BPE.tok_config`.

To build separate models for source and target sides, run the same command using as input only source (or target) data.

* Build Vocabularies:

```
cat $TRAIN.$SS | python3 buildVOC_cli.py -tok_config $BPE.tok_config > $VOC.$SS
cat $TRAIN.$TT | python3 buildVOC_cli.py -tok_config $BPE.tok_config > $VOC.$TT
```

Vocabularies are computed after tokenizing files following `$BPE.tok_config`. 

### (2) Create network

```
python3 ./create_cli.py -dnet $DNET -src_vocab $VOC.$SS -tgt_vocab $VOC.$TT -src_token $BPE.tok -tgt_token $BPE.tok
```

Creates the directory `$DNET` with the next files: 
* Network description: 
  * network
* Copies vocabularies, tokenization options and BPE models:
  * src_voc
  * tgt_voc 
  * src_tok 
  * tgt_tok 
  * src_bpe 
  * tgt_bpe

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



