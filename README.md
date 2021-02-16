# Minimalistic implementation of a NMT toolkit using Transformers

A framework developed for research purposes, aiming to be a clean and minimalistic code base achieving similar accuracy than other state-of-the art frameworks.

The framework is built on/employs:
* PyTorch (https://pytorch.org)
* Google SentencePiece (https://github.com/google/sentencepiece)
* TensorBoard visualizations

## Clients

Preprocessing:
* `tools/spm_train.py` : Learns a SentencePiece model over raw (untokenized) text files

Network:
* `minmt-setup.py` : Creates the NMT network experiment
* `minmt-train.py` : Runs learning 
* `minmt-translate.py`: Runs inference

Run clients with the -h option for a detailed description of available options.

## Usage example:

Hereinafter we consider `train.en, train.fr`, `valid.en, valid.fr` and `test.en` the respective train/valid/test files of our example.
All files are formated with one sentence per line of untokenized (raw) text.

### (1) Preprocessing

* Build a tokenization (SentencePiece) model and vocabulary:
```
python3 tools/spm_train.py -sp_model SP_enfr -i train.{en,fr}
```
The script outputs `SP_enfr.model` and `SP_enfr.vocab` files for both, source and target, sides of parallel data. 
By default, the vocabulary contains the 30,000 most frequent words. The vocabulary is not further needed, already contained in the model file.

You can use separate SentencePiece models (vocabularies) for each translation side:
```
python3 tools/spm_train.py -sp_model SP_en -i train.en
python3 tools/spm_train.py -sp_model SP_fr -i train.fr
```

Thus producing `SP_en.model`, `SP_en.vocab`, `SP_fr.model` and `SP_fr.vocab`.

You can use the original spm_train binary (https://github.com/google/sentencepiece) with your preferred options making sure that you set: `--pad_id=0`, `--pad_piece='<pad>'`, `--unk_id=1`, `--unk_piece='<unk>'`, `--bos_id=2`, `--bos_piece='<bos>'`, `--eos_id=3`, `--eos_piece='<eos>'`.

### (2) Create the network


If you built a single model/vocabulary:
```
python3 minmt-setup.py -dnet $DNET -src_spm SP_enfr.model -tgt_spm SP_enfr.model
```

Otherwise:
```
python3 minmt-setup.py -dnet $DNET -src_spm SP_en.model -tgt_spm SP_fr.model
```

The script creates the directory `$DNET` containing:
* network (the network description file)
* src_spm (source-side SentencePiece model)
* tgt_spm (target-side SentencePiece model)

(source and target vocabularies are contained in SentencePiece models)

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

To translate an input file (using the last network checkpoint), run the command:
```
python3 minmt-translate.py -dnet $DNET -i test.en
```

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
[i] index of sentence in test set
[n] rank in n-best
[c] global hypothesis cost
[s] input sentence (tokenised)
[S] input sentence (raw)
[u] input sentence (indexes)
[h] hypothesis (tokenised)
[H] hypothesis (raw)
[v] hypothesis (indexes)
```

Same as Train/Validation datasets, test datasets are handled using `src_spm` and `tgt_spm` SentencePiece models existing in `$DNET` directory.


## Use of GPU:

It is highly recommended to use a GPU for training/inference. If you have one, you can prefix the training/inference commands with the `CUDA_VISIBLE_DEVICES=i` envoronment variable as well as adding the `-cuda` option. Ex:

```
CUDA_VISIBLE_DEVICES=0 python3 minmt-train.py -dnet $DNET -src_train train.en -tgt_train train.fr -src_valid valid.en -tgt_valid valid.fr -cuda
CUDA_VISIBLE_DEVICES=0 python3 minmt-translate.py -dnet $DNET -i test.en -cuda
```


