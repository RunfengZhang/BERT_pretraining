#! /bin/bash

python preprocess_pinyin.py \
    --model_name_or_path bert-base-chinese \
    --preprocessing_num_workers 100 \
    --max_seq_length 128 \
    --cache_dir ./cache/ \
    --output_dir ./output-pinyin \
    --train_file /home/web_server/antispam/project/wangkai/review-pretrain/review_data/comment_merge_0708_0711_tfidf_samll.txt \
    --data_dir ./review_data/pinyin_tokenized_data
