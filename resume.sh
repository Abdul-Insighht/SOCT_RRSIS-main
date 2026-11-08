#!/bin/bash
# ==============================================================================
#                 SOCT-SAM GPU Resume Training Script
# ==============================================================================
# Resume training from checkpoint with fine-tuning settings.
#
# Usage:
#   bash resume.sh [dataset_name] [data_root] [sam3_ckpt]
#
# Examples:
#   bash resume.sh rrsis_d ./data ./pre-trained-weights/sam3.pt
# ==============================================================================

# Default parameters (matching fine.sh exactly)
DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
SAM3_CKPT=${3:-./pre-trained-weights/sam3.pt}
OUTPUT_DIR="./output/${DATASET}_soct_sam"
RESUME_CKPT=${4:-"${OUTPUT_DIR}/latest_model.pth"}
USER_IMG_SIZE=${5:-""}

# Auto-detect checkpoint if default does not exist
if [ ! -f "${RESUME_CKPT}" ] && [ ! -d "${RESUME_CKPT}" ]; then
    for cand in "${OUTPUT_DIR}/best_model.pth" "./checkpoints/best_model.pth"; do
        if [ -f "${cand}" ] || [ -d "${cand}" ]; then
            RESUME_CKPT="${cand}"
            break
        fi
    done
fi

# Determine mathematically optimal patch-14 image size
if [ -n "${USER_IMG_SIZE}" ]; then
    IMG_SIZE=${USER_IMG_SIZE}
elif [ "${DATASET}" == "rrsis_hr" ]; then
    IMG_SIZE=784
else
    IMG_SIZE=504
fi

# Set safe batch size and gradient accumulation to prevent CUDA OOM
if [ "${IMG_SIZE}" -gt 600 ]; then
    BATCH_SIZE=2
    GRAD_ACCUM=8
else
    BATCH_SIZE=4
    GRAD_ACCUM=4
fi

# Highlight setup information
echo "=============================================================================="
echo "🔄 Resuming SOCT-SAM training on GPU..."
echo "📊 Dataset:         ${DATASET}"
echo "📂 Data Root:       ${DATA_ROOT}"
echo "💾 SAM3 Ckpt:       ${SAM3_CKPT}"
echo "📁 Output Dir:      ${OUTPUT_DIR}"
echo "💾 Resume Ckpt:     ${RESUME_CKPT}"
echo "🖼️  Image Size:      ${IMG_SIZE} (divisible by patch size 14)"
echo "📦 Batch / Accum:   ${BATCH_SIZE} x ${GRAD_ACCUM} (Effective BS = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "🔧 Optimizations:   Cosine LR Floor | LoRA Dropout 0.10 | IoU Query Matching"
echo "=============================================================================="

# Ensure the resume checkpoint exists
if [ ! -f "${RESUME_CKPT}" ] && [ ! -d "${RESUME_CKPT}" ]; then
    echo "[ERROR] Could not find the resume checkpoint file or directory at '${RESUME_CKPT}'!"
    exit 1
fi

# Run the training script with optimal hyperparameters for performance & speed
python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ${SAM3_CKPT} \
    --resume ${RESUME_CKPT} \
    --image_size ${IMG_SIZE} \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 30 \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --lr 2e-5 \
    --lr_backbone 5e-6 \
    --lr_decoder 2e-5 \
    --weight_decay 0.03 \
    --warmup_epochs 2 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    --contrastive_weight 0.1 \
    --ohem_hard_ratio 0.3 \
    --ot_reg 0.1 \
    --ot_num_iter 10 \
    --num_ot_scales 3 \
    --use_soct \
    --lambda_ord 5.0 \
    --lambda_occ 1.0 \
    --lambda_rho 0.5 \
    --focal_gamma 2.0

echo "Resumed training epoch run complete!"
