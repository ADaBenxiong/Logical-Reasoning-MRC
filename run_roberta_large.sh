export RECLOR=reclor_data
export TASK_NAME=reclor
export MODEL_NAME=roberta-large-test
#export MODEL_NAME=albert-xxlarge-v2-test

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model_type $MODEL_NAME \
    --task_name $TASK_NAME \
    --data_dir $RECLOR \
    --model_name_or_path $MODEL_NAME  \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --do_test \
    --do_fgm \
    --do_lower_case \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 24  \
    --per_gpu_train_batch_size 1   \
    --gradient_accumulation_steps 24 \
    --learning_rate 1e-05 \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${MODEL_NAME}/newGCN_ver \
    --logging_steps 100 \
    --save_steps 1000 \
    --use_pool \
    --overwrite_output_dir