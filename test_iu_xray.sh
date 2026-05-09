MODEL_NAME="${MODEL_NAME:-}"
DATASET_PATH="${DATASET_PATH:-data/iu_xray}"

python main_test.py \
    --image_dir "$DATASET_PATH/iu_xray/images/" \
    --ann_path "$DATASET_PATH/iu_xray/annotation.json" \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --save_dir results/iu_xray \
    --step_size 50 \
    --gamma 0.1 \
    --seed 9223 \
    --load "$MODEL_NAME" \
