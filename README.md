# Basic Transformer implementation for Machine Translation

Clients:
* `learnBPE_cli` : Learns BPE model after applying tokenization ("aggressive", joiner_annotate=True, segment_numbers=True)
* `buildVOC_cli` : Builds NMT vocabulary given tokenization
* `tokenTXT_cli` : Tokenizes raw data
* `buildIDX_cli` : Converts raw data to corresponding idx given tokenization and vocabularies

Network:
* `create_cli` : Creates new network
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference
