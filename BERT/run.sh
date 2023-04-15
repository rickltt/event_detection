#!/bin/bash

MODEL_NAME_OR_PATH='bert-base-uncased'
OUTPUT_DIR='./output'

CUDA_VISIBLE_DEVICES='1' python run.py \
    --dataset ace \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --evaluate_during_training \
    --evaluate_after_epoch \
    --per_gpu_train_batch_size 24 \
    --per_gpu_eval_batch_size 24 \
    --dropout_prob 0.1 \
    --max_seq_length 256 \
    --logging_steps -1 \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --seed 42