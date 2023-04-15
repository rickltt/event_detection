#!/bin/bash
OUTPUT_DIR='./output'
CUDA_VISIBLE_DEVICES='0' python run.py \
    --seed 112 \
    --dataset ace \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 256 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 128 \
    --dropout_prob 0.3 \
    --learning_rate 3e-4 \
    --weight_decay 1e-8 \
    --num_train_epochs 50 \
    --hidden_size 300 \
    --rnn_hidden 256 \
    --num_layers 1 \
    --do_train \
    --do_eval \
    --output_dir output \
    --evaluate_during_training \
    --evaluate_after_epoch

