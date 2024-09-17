TENSORBOARD_DIR=/home/web_server/antispam/project/wangkai/tensorboard/pretrain

TASK=politic_pinyin_pretrain_10w

PYTHONPATH=../ python -m torch.distributed.launch --master_port 2951 --nproc_per_node=8 finetune_politic.py --weight_decay=0.01 --num_train_epochs=3 --output_dir=ckpt/$TASK --log_file=logs/$TASK.log --tensorboard_dir=$TENSORBOARD_DIR/$TASK --per_device_train_batch_size=32 --warmup_ratio=0.06 --learning_rate=5e-5
