#! /bin/bash
torchrun --nnodes=3 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=11.39.106.93:29400 mlm.py \
    --model_name_or_path ./output-mlm-bert-large/checkpoint-460000 \
    --tokenizer_name kwai_tokenizer \
    --per_device_train_batch_size 43 \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps 4 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --do_train \
    --data_dir ./text_pretrain_data \
    --report_to tensorboard \
    --logging_dir /home/web_server/antispam/project/wangkai/tensorboard/pretrain_mlm_bert_large \
    --cache_dir ./cache/ \
    --output_dir ./output-mlm-bert-large-continue \
    --num_train_epochs 50 \
    --optim  adamw_hf \
    --seed 42 \
    --fp16 true \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --adam_beta2 0.98 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --ignore_data_skip

