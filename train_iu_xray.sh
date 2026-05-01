#! /bin/bash

#  --- CHANGE EXP PARAMS HERE, ONLY THIS FILE! --- 
WANDB_ENTITY="phucga15062005" # team wandb  --> NOT CHANGE!
WANDB_PROJECT="NCKH_R2Gen" # Project wandb --> NOT CHANGE!
# Lấy WANDB_NAME từ environment (nếu được set trên Kaggle), ngược lại dùng tên mặc định
WANDB_NAME="${WANDB_NAME:-a3net_graphlite_alpha05_warmup10-$(date '+%Y%m%d-%H%M')}"

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
USE_REGION_PROMPTS="${USE_REGION_PROMPTS:-false}"
NUM_REGION_PROMPTS="${NUM_REGION_PROMPTS:-8}"
USE_WEIGHTED_NLL="${USE_WEIGHTED_NLL:-1}"  # 0: baseline Masked NLL, 1: GraphLite Weighted Masked NLL
WEIGHTED_NLL_ALPHA="${WEIGHTED_NLL_ALPHA:-0.5}"
WEIGHTED_NLL_WARMUP_EPOCHS="${WEIGHTED_NLL_WARMUP_EPOCHS:-10}"

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
    --use_weighted_nll "$USE_WEIGHTED_NLL" \
    --weighted_nll_alpha "$WEIGHTED_NLL_ALPHA" \
    --weighted_nll_warmup_epochs "$WEIGHTED_NLL_WARMUP_EPOCHS" \
    --optim AdamW \
    --weight_decay 1e-4\
    --lr_scheduler WarmupCosine \
    --warmup_epochs 5 \
    --warmup_start_factor 0.1 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --swin_num_layers "${SWIN_NUM_LAYERS:-3}" \
    --decoder_num_layers "${DECODER_NUM_LAYERS:-3}" \
    --use_view_type_embedding "$USE_VIEW_TYPE_EMBEDDING" \
    --use_region_prompts "$USE_REGION_PROMPTS" \
    --num_region_prompts "$NUM_REGION_PROMPTS" \
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
