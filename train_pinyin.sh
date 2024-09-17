#! /bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr="10.82.40.90" --master_port 2950 --nproc_per_node=8 run_mlm_pinyin.py \
    --model_name_or_path output-pinyin \
    --per_device_train_batch_size 120 \
    --preprocessing_num_workers 20 \
    --gradient_accumulation_steps 5 \
    --save_steps 2000 \
    --save_total_limit 20 \
    --do_train \
    --data_dir ./review_data/pinyin_tokenized_data \
    --cache_dir ./cache/ \
    --output_dir ./output-pinyin \
    --num_train_epochs 200 \
    --overwrite_output_dir \
    --remove_unused_columns False
