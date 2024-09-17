#! /bin/bash

torchrun  --nproc_per_node=8 wwm_mlm.py \
    --model_name_or_path bert-base-chinese \
    --per_device_train_batch_size 300 \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps 5 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --do_train \
    --data_dir ./review_data/line_by_line_tokenized_data \
    --cache_dir ./cache/ \
    --output_dir ./output-mac \
    --num_train_epochs 5 \
    --overwrite_output_dir \
    --remove_unused_columns False
