#!/usr/bin/env bash

nproc_per_node=1
NPROC_PER_NODE=$nproc_per_node \
USE_HF=1 \
swift sft \
    --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --train_type lora \
    --dataset 'dataset/train.json' \
    --torch_dtype float32 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 4 \
    --save_steps 2 \
    --eval_strategy steps \
    --use_chat_template false \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --learning_rate 5e-6 \
    --dataloader_num_workers 4 \
    --task_type embedding \
    --loss_type cosine_similarity \
    --dataloader_drop_last true
