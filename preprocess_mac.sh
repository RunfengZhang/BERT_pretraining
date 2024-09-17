#! /bin/bash

python preprocess_mac.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --preprocessing_num_workers 100 \
    --data_dir ./review_data/wwm_tokenized_data \
    --cache_dir ./cache/ \
    --output_dir ./output \
    --train_file /home/web_server/antispam/project/chengang06/data/comment_data/2102.10-15.24-01.comment.clean.uni.txt \
    --max_seq_length 512