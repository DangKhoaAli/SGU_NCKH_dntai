#! /bin/bash

#  --- CHANGE EXP PARAMS HERE, ONLY THIS FILE! --- 
WANDB_ENTITY="phucga15062005" # team wandb  --> NOT CHANGE!
WANDB_PROJECT="NCKH_R2Gen" # Project wandb --> NOT CHANGE!
# Lấy WANDB_NAME từ environment (nếu được set trên Kaggle), ngược lại dùng tên mặc định
WANDB_NAME="${WANDB_NAME:-team-scratch-AdamW-ReduceLROnPlateau-$(date '+%Y%m%d-%H%M')}"

RESUME_PATH="${RESUME_PATH:-}"

# update for dataset path in kaggle IU_XRAY_RRG
# đọc DATASET_PATH từ environment (set trên kaggle), nếu không có thì dùng local path
DATASET_PATH="${DATASET_PATH:-data/iu_xray}"
N_GPU="${N_GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

# Giam memory fragmentation tren CUDA
export PYTORCH_ALLOC_CONF=expandable_segments:True

python main_train.py\
    --image_dir "$DATASET_PATH/iu_xray/images/" \
    --ann_path "$DATASET_PATH/iu_xray/annotation.json" \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --n_gpu "$N_GPU" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --accum_steps "$ACCUM_STEPS" \
    --use_amp \
    --optim AdamW \
    --lr_scheduler ReduceLROnPlateau \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --reduce_on_plateau_factor 0.5 \
    --reduce_on_plateau_patience 5 \
    --reduce_on_plateau_threshold 1e-4 \
    --min_lr 1e-6 \
    --num_layers 3 \
    --early_stop 50 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir "results/iu_xray_$(date '+%d-%m-%Y_%H%M')/" \
    --log_period 50 \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    ${RESUME_PATH:+--resume "$RESUME_PATH"}
