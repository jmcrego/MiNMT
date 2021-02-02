# Basic Transformer implementation for MT

## Clients

Preprocess:
* `learnBPE_cli` : Learns BPE model after applying tokenization ("aggressive", joiner_annotate=True, segment_numbers=True)
* `buildVOC_cli` : Builds vocabulary given tokenization
* `tokenTXT_cli` : Tokenizes raw data
* `buildIDX_cli` : Builds batches from raw data given tokenization and vocabularies

Network:
* `create_cli` : Creates network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference

## Usage example:

Given train/valid/test datasets:

### (1) Preprocess

Build `$BPE` Model:

```
cat $TRAIN.{$SS,$TT} | python3 learnBPE_cli.py $BPE
```

Create tokenization config file `$TOK` containing:

```
mode: aggressive
joiner_annotate: True
segment_numbers: True
bpe_model_path: $BPE
```

Build Vocabularies:

```
cat $TRAIN.$SS | python3 buildVOC_cli.py -tokenizer_config $TOK -max_size 32768 > $VOC.$SS
cat $TRAIN.$TT | python3 buildVOC_cli.py -tokenizer_config $TOK -max_size 32768 > $VOC.$TT
```

### (2) Create network

```
python3 ./create_cli.py -dnet $DNET -src_vocab $VOC.$SS -tgt_vocab $VOC.$TT -src_token $TOK -tgt_token $TOK
```

### (3) Learning
```
python3 ./train_cli.py -dnet $DNET -src_train $TRAIN.$SS -tgt_train $TRAIN.$TT -src_valid $VALID.$SS -tgt_valid $VALID.$TT
```

### (4) Inference
```
python3 ./translate_cli.py -dnet $DNET -i $TEST.$SS
```


