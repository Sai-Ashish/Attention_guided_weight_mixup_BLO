#!/bin/bash

GPU=0
TASK_NAME=rte
TRAIN_SPLIT=-1
MODEL=bert-large-cased
BSZ=16
EPOCH=1.0
LR=2e-5
MLO_EPOCHS=3
MLO_WARMUP=0
MODEL_LR=2e-5
ALPHA_LR=2e-3
ALPHA_WARMUP_RATIO=0.1
ALPHA_WEIGHT_DECAY=0
USE_L1=false
ALPHA_L1_FACTOR=0
UNROLL_STEPS=1
AVG=true

for SEED in 1017 #1017 4803 6757 9618 10290 11799 19189 28104 29576 31511
do
    CUDA_VISIBLE_DEVICES=${GPU} python run_glue_mlo.py \
        --model_name_or_path ${MODEL} \
        --do_train \
        --do_eval \
        --task_name ${TASK_NAME} \
        --max_seq_length 128 \
        --seed ${SEED} \
        --per_device_train_batch_size ${BSZ} \
        --learning_rate ${LR} \
        --num_train_epochs ${EPOCH} \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --output_dir ./Output/run_${SEED}_${TASK_NAME} \
        --train_split ${TRAIN_SPLIT} \
        --use_mlo true \
        --mlo_sample_dataset true \
        --save_total_limit 1 \
        --save_steps 30000 \
        --overwrite_output_dir \
        --MLO_warm_up ${MLO_WARMUP} \
        --MLO_epochs ${MLO_EPOCHS} \
        --unroll_steps ${UNROLL_STEPS} \
        --model_learning_rate ${MODEL_LR} \
        --alpha_learning_rate ${ALPHA_LR} \
        --alpha_warmup_ratio ${ALPHA_WARMUP_RATIO} \
        --alpha_weight_decay ${ALPHA_WEIGHT_DECAY} \
        --use_l1 ${USE_L1} \
        --L1factor ${ALPHA_L1_FACTOR} \
        --exp_name ${SEED}_${TASK_NAME} \
        --report_freq 100 \
        --alpha_lr_scheduler_type cosine \
        --cross_valid 5.0 \
        --total_avg ${AVG}
done