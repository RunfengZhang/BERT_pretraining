#! /bin/bash

torchrun  --nproc_per_node=8 mlm.py \
    --config_name junnyu/roformer_chinese_base \
    --tokenizer_name kwai_tokenizer \
    --per_device_train_batch_size 250 \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps 5 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --do_train \
    --data_dir ./text_pretrain_data \
    --report_to tensorboard \
    --logging_dir /home/web_server/antispam/project/wangkai/tensorboard/pretrain_mlm \
    --cache_dir ./cache/ \
    --output_dir ./output-mlm \
    --num_train_epochs 50 \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --optim  adamw_hf \
    --seed 42 \
    --fp16 true \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --warmup_steps 25000 \
    --adam_beta2 0.98 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine


