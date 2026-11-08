#!/bin/bash
# ====== SOCT-SAM Evaluation Script ======
#
# Usage:
#   bash test.sh <dataset_name> <split> <checkpoint_path>
#   e.g., bash test.sh rrsis_d test ./output/rrsis_d_soct_sam/best_model.pth

DATASET=${1:-rrsis_d}
SPLIT=${2:-test}
CKPT=${3:-./output/${DATASET}_soct_sam/best_model.pth}
DATA_ROOT=${4:-./data}
USER_IMG_SIZE=${5:-""}

# Auto-detect optimal patch-14 image size
if [ -n "${USER_IMG_SIZE}" ]; then
    IMG_SIZE=${USER_IMG_SIZE}
elif [ "${DATASET}" == "rrsis_hr" ]; then
    IMG_SIZE=784
else
    IMG_SIZE=504
fi

echo "============================================="
echo "  SOCT-SAM Evaluation"
echo "  Dataset:    ${DATASET}"
echo "  Split:      ${SPLIT}"
echo "  Checkpoint: ${CKPT}"
echo "  Image Size: ${IMG_SIZE}"
echo "============================================="

python test.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --split ${SPLIT} \
    --resume ${CKPT} \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size ${IMG_SIZE} \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --use_soct \
    --eval_only \
    --visualize \
    --num_workers 4
