#! /bin/bash
export http_proxy=http://10.28.121.13:11080 https_proxy=http://10.28.121.13:11080

python -m torch.distributed.launch --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port 2950 --nproc_per_node=8 sample_mlm.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --per_device_train_batch_size 28 \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps 20 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --do_train \
    --data_dir ./review_data/chinese_tokenized_data/data_with_freq \
    --cache_dir ./cache/ \
    --output_dir ./output-sample-wwm \
    --num_train_epochs 5 \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --logging_dir /home/web_server/antispam/project/wangkai/tensorboard