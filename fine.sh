#!/bin/bash
# ==============================================================================
#                 SOCT-SAM GPU Training Script
# ==============================================================================
# Highly optimized training launcher for SOCT-SAM on GPU.
# Applies SOCT alignment, Dynamic LoRA, Contrastive Loss, and OHEM Loss.
#
# Usage:
#   bash fine.sh [dataset_name] [data_root] [sam3_ckpt]
#
# Examples:
#   bash fine.sh rrsis_d ./data ./pre-trained-weights/sam3.pt
# ==============================================================================

# Default parameters
DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
SAM3_CKPT=${3:-./pre-trained-weights/sam3.pt}
USER_IMG_SIZE=${4:-""}
OUTPUT_DIR="./output/${DATASET}_soct_sam"

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
echo "🚀 Starting SOCT-SAM training on GPU..."
echo "📊 Dataset:         ${DATASET}"
echo "📂 Data Root:       ${DATA_ROOT}"
echo "💾 SAM3 Ckpt:       ${SAM3_CKPT}"
echo "📁 Output Dir:      ${OUTPUT_DIR}"
echo "🖼️  Image Size:      ${IMG_SIZE} (divisible by patch size 14)"
echo "📦 Batch / Accum:   ${BATCH_SIZE} x ${GRAD_ACCUM} (Effective BS = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "🔧 Optimizations:   Dynamic LoRA | Contrastive Loss | SOCT Aligner | OHEM Loss"
echo "=============================================================================="

# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

# Ensure pre-trained weight exists or check parent dir fallback
if [ ! -f "${SAM3_CKPT}" ]; then
    echo "⚠️  [WARNING] Pre-trained weight not found at '${SAM3_CKPT}'"
    FALLBACK="../sam3_pre_trained_weights/sam3.pt"
    if [ -f "${FALLBACK}" ]; then
        echo "🔍 [INFO] Found fallback checkpoint in parent directory: '${FALLBACK}'"
        SAM3_CKPT=${FALLBACK}
    else
        echo "❌ [ERROR] Could not find pre-trained weights. Please make sure sam3.pt is placed correctly."
    fi
fi

# Run the training script with optimal hyperparameters
python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ${SAM3_CKPT} \
    --image_size ${IMG_SIZE} \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 30 \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
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

echo "🎉 Training epoch run complete!"
