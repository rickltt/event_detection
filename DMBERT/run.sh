#!/bin/bash
MODEL_NAME_OR_PATH='/code/ltt_code/bert/bert-base-uncased'
OUTPUT_DIR='./ace_output'
DATASET='ace'
CUDA_VISIBLE_DEVICES='1' python run.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_seq_length 256 \
    --per_gpu_train_batch_size 42 \
    --per_gpu_eval_batch_size 42 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 1e-5 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training