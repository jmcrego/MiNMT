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

Given training/validation/test datasets:

### 1 Preprocess

Build `$fBPE` Model:
```cat $TRAINING.{$SS,$TT} | python3 learnBPE_cli.py $fBPE```

Create tokenization config file `$fTOK`:

```
mode: aggressive
joiner_annotate: True
segment_numbers: True
bpe_model_path: $fBPE
```

Build Vocabularies:
`cat $TRAINING.$SS | python3 buildVOC_cli.py -tokenizer_config $fTOK -max_size 32768 > $VOC.$SS`
`cat $TRAINING.$TT | python3 buildVOC_cli.py -tokenizer_config $fTOK -max_size 32768 > $VOC.$TT`


