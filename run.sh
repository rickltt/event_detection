 #! /bin/bash
dataset='maven'
model_type='bert'
model_name_or_path='/home/ltt/ltt_code/bert/bert-large-uncased'
batch_size=8
max_len=256
epoch=5
lr=3e-5
prompt_list=(p1 p2 p3 p4)
for p in ${prompt_list[@]}
do
    python test.py \
        --dataset $dataset \
        --model_type $model_type \
        --model_name_or_path $model_name_or_path \
        --prompt $p \
        --max_seq_length $max_len \
        --batch_size $batch_size \
        --num_train_epochs $epoch \
        --learning_rate $lr
done