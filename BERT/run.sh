#!/bin/bash

MODEL_NAME_OR_PATH='/code/ltt_code/bert/bert-base-uncased'
OUTPUT_DIR='./ace_output'

CUDA_VISIBLE_DEVICES='2' python run.py \
    --dataset ace \
    --do_train \
    --do_eval \
    --use_crf \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --evaluate_after_epoch \
    --per_gpu_train_batch_size 42 \
    --per_gpu_eval_batch_size 42 \
    --dropout_prob 0.1 \
    --max_seq_length 256 \
    --learning_rate 3e-5 \
    --weight_decay 5e-5 \
    --num_train_epochs 10 \
    --seed 42