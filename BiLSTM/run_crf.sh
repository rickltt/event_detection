#!/bin/bash
OUTPUT_DIR='./maven_crf_output'
CUDA_VISIBLE_DEVICES='2' python run.py \
    --seed 42 \
    --dataset maven \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --dropout_prob 0.3 \
    --learning_rate 3e-4 \
    --weight_decay 1e-8 \
    --num_train_epochs 100 \
    --hidden_size 100 \
    --rnn_hidden 256 \
    --num_layers 1 \
    --do_train \
    --do_eval \
    --use_crf \
    --evaluate_during_training \
    --evaluate_after_epoch

