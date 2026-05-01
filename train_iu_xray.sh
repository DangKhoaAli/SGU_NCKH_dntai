#! /bin/bash

#  --- CHANGE EXP PARAMS HERE, ONLY THIS FILE! --- 
WANDB_ENTITY="phucga15062005" # team wandb  --> NOT CHANGE!
WANDB_PROJECT="NCKH_R2Gen" # Project wandb --> NOT CHANGE!
# Lấy WANDB_NAME từ environment (nếu được set trên Kaggle), ngược lại dùng tên mặc định
WANDB_NAME="${WANDB_NAME:-team-scratch-AdamW-StepLR-$(date '+%Y%m%d-%H%M')}"

RESUME_PATH="${RESUME_PATH:-}"

# update for dataset path in kaggle IU_XRAY_RRG
# đọc DATASET_PATH từ environment (set trên kaggle), nếu không có thì dùng local path
DATASET_PATH="${DATASET_PATH:-data/iu_xray}"
N_GPU="${N_GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
USE_IU_XRAY_VIEW_FILTER="${USE_IU_XRAY_VIEW_FILTER:-true}"
IU_XRAY_VIEW_FILTER_CSV="${IU_XRAY_VIEW_FILTER_CSV:-/kaggle/input/datasets/quooccuongwf/dataset-errors/iu_xray_select_2views_by_cosine.csv}"
USE_VIEW_TYPE_EMBEDDING="${USE_VIEW_TYPE_EMBEDDING:-false}"

# Giam memory fragmentation tren CUDA
export PYTORCH_ALLOC_CONF=expandable_segments:True

python main_train.py\
    --image_dir "$DATASET_PATH/iu_xray/images/" \
    --ann_path "$DATASET_PATH/iu_xray/annotation.json" \
    --dataset_name iu_xray \
    --use_iu_xray_view_filter "$USE_IU_XRAY_VIEW_FILTER" \
    --iu_xray_view_filter_csv "$IU_XRAY_VIEW_FILTER_CSV" \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --n_gpu "$N_GPU" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --accum_steps "$ACCUM_STEPS" \
    --use_amp \
    --optim AdamW \
    --visual_extractor "${VISUAL_EXTRACTOR:-densenet121}" \
    --d_vf "${D_VF:-1024}" \
    --lr_scheduler ReduceLROnPlateau \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --reduce_on_plateau_factor 0.5 \
    --reduce_on_plateau_patience 3 \
    --reduce_on_plateau_threshold 1e-4 \
    --reduce_on_plateau_cooldown 0 \
    --num_layers 3 \
    --swin_num_layers "${SWIN_NUM_LAYERS:-3}" \
    --decoder_num_layers "${DECODER_NUM_LAYERS:-4}" \
    --drop_prob_lm "${DROP_PROB_LM:-0.35}" \
    --use_view_type_embedding "$USE_VIEW_TYPE_EMBEDDING" \
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
