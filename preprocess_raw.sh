#! /bin/bash

python preprocess_raw.py \
    --tokenizer_name kwai_tokenizer \
    --preprocessing_num_workers 100 \
    --data_dir ./text_pretrain_data/photo_searchword_concat_ab \
    --cache_dir ./cache/ \
    --output_dir ./output \
    --train_file data/photo_searchword_concat_ab.txt \
    --max_seq_length 256