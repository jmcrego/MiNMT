# Transformer

Neural Machine Translation Network based on Transformer

Clients:
* '''learnBPE_cli''' : Learns BPE model after applying tokenization ("aggressive", joiner_annotate=True, segment_numbers=True)
* buildVOC_cli : Builds NMT vocabulary given tokenization
* tokenTXT_cli : Tokenizes raw data
* buildIDX_cli : Converts raw data to corresponding idx given tokenization and vocabularies

Network:
* net_create_cli
* net_learn_cli
* net_translate_cli
