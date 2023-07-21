#!/bin/bash
OUTPUT_DIR='./maven_output'
CUDA_VISIBLE_DEVICES='4' python run.py \
    --seed 42 \
    --dataset maven \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --dropout_prob 0.1 \
    --learning_rate 2e-4 \
    --num_train_epochs 50 \
    --rnn_hidden 300 \
    --do_train \
    --do_eval \

