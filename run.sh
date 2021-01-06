#!/bin/bash

data(){
    cat /nfs/CORPUS/aligned/news-commentary-v14_enfr-checked2_lo.en /nfs/CORPUS/aligned/europarl-v8_enfr-checked2.en > $dir/train.en
    cat /nfs/CORPUS/aligned/news-commentary-v14_enfr-checked2_lo.fr /nfs/CORPUS/aligned/europarl-v8_enfr-checked2.fr > $dir/train.fr
    cat files/newstest2008-enfr.en.txt > $dir/valid.en
    cat files/newstest2008-enfr.fr.txt > $dir/valid.fr
    cat files/newstest2009-enfr.en.txt > $dir/test.en
    cat files/newstest2009-enfr.fr.txt > $dir/test.fr
}

preprocess(){
    echo -e "mode: aggressive\njoiner_annotate: True\nsegment_numbers: True\nbpe_model_path: $fbpe" > $ftok
    cat $dir/train.{en,fr} | python3 ./learnBPE_cli.py $fbpe #joint bpe
    cat $dir/train.en | python3 buildvoc_cli.py -tokenizer_config $ftok -max_size 32768 > $voc_ss
    cat $dir/train.fr | python3 buildvoc_cli.py -tokenizer_config $ftok -max_size 32768 > $voc_tt
    python3 ./word2idx_cli.py -src $dir/valid.en -tgt $dir/valid.fr -src_vocab $voc_ss -tgt_vocab $voc_tt -src_token $ftok -tgt_token $ftok -set $valid
    python3 ./word2idx_cli.py -src $dir/train.en -tgt $dir/train.fr -src_vocab $voc_ss -tgt_vocab $voc_tt -src_token $ftok -tgt_token $ftok -set $train
    python3 ./word2idx_cli.py -src $dir/test.en -src_vocab $voc_ss -src_token $ftok -set $test
}

train(){
    CUDA_VISIBLE_DEVICES=0 python3 ./Transformer.py -suffix $suffix -src_vocab $voc_ss -tgt_vocab $voc_tt -train_set $train -valid_set $valid -batch_size 32 -batch_type sentences -max_length 100 -report_every 100 -save_every 5000 -cuda -log_file $suffix.trn.log &
}

inference(){
    python3 ./Transformer.py -suffix $suffix -src_vocab $voc_ss -tgt_vocab $voc_tt -test_set $test -batch_size 3 -batch_type sentences -beam_size 4 -n_best 2 -max_size 15 
}

dir=$PWD/files
mkdir -p $dir/bin
fbpe=$dir/bin/enfr.32k.bpe
ftok=$dir/bin/tokconf
voc_ss=$dir/bin/en.vocab
voc_tt=$dir/bin/fr.vocab
suffix=$dir/bin/model
train=$dir/bin/train-enfr.bin
valid=$dir/bin/valid-enfr.bin
test=$dir/bin/test-en.bin

#data
#preprocess
#train
#inference
