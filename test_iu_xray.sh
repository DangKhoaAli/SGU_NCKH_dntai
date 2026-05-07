N_GPU="${N_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_IU_XRAY_VIEW_FILTER="${USE_IU_XRAY_VIEW_FILTER:-true}"
IU_XRAY_VIEW_FILTER_CSV="${IU_XRAY_VIEW_FILTER_CSV:-/kaggle/input/datasets/quooccuongwf/dataset-errors/iu_xray_select_2views_by_cosine.csv}"


# -----------------------------------------------------------------------
# Chế độ dịch mẫu nhanh (demo) — chỉ set 1 trong 3 biến bên dưới.
# Nếu không set biến nào → chạy full test + tính metrics như bình thường.
#
#   DEMO_FOLDER_IDS : tên folder(s) chứa ảnh (ưu tiên cao nhất)
#                     Ví dụ: DEMO_FOLDER_IDS="CXR1_1_IM-0001 CXR3_1_IM-0005"
#   DEMO_N          : dịch N mẫu đầu tiên trong test set
#                     Ví dụ: DEMO_N=5
#   DEMO_INDICES    : dịch các mẫu theo index trong test set
#                     Ví dụ: DEMO_INDICES="0 3 7"
# -----------------------------------------------------------------------
DEMO_FOLDER_IDS="${DEMO_FOLDER_IDS:-}"
DEMO_N="${DEMO_N:-}"
DEMO_INDICES="${DEMO_INDICES:-}"
MODEL_NAME="${MODEL_NAME:-}"
python main_test.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --use_iu_xray_view_filter "$USE_IU_XRAY_VIEW_FILTER" \
    --iu_xray_view_filter_csv "$IU_XRAY_VIEW_FILTER_CSV" \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --n_gpu "$N_GPU" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --swin_num_layers "${SWIN_NUM_LAYERS:-5}" \
    --decoder_num_layers "${DECODER_NUM_LAYERS:-3}" \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray/ \
    --log_period 50 \
    --load "$MODEL_NAME" \
    ${DEMO_FOLDER_IDS:+--demo_folder_ids $DEMO_FOLDER_IDS} \
    ${DEMO_N:+--demo_n $DEMO_N} \
    ${DEMO_INDICES:+--demo_indices $DEMO_INDICES}
