#!/bin/bash
MODEL_NAME_OR_PATH='bert-base-uncased'
OUTPUT_DIR='./output'
DATASET='ace'
CUDA_VISIBLE_DEVICES='2' python run.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 42 \
    --per_gpu_eval_batch_size 42 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training