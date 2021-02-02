# Basic Transformer implementation for MT

Preprocess:
* `learnBPE_cli` : Learns BPE model after applying tokenization ("aggressive", joiner_annotate=True, segment_numbers=True)
* `buildVOC_cli` : Builds vocabulary given tokenization
* `tokenTXT_cli` : Tokenizes raw data
* `buildIDX_cli` : Builds batches from raw data given tokenization and vocabularies

Network:
* `create_cli` : Creates new network from scratch
* `learn_cli` : Runs learning 
* `translate_cli`: Runs inference
