"""
Threshold Sweep Script for SOCT-SAM
====================================
Tests multiple mask binarization thresholds on the val split
to find the optimal threshold that maximizes mIoU and oIoU.

Usage:
    python threshold_sweep.py \
        --dataset rrsis_d \
        --data_root ./data/ \
        --resume ./output/rrsis_d_784_scratch/best_model.pth \
        --image_size 784 \
        --lora_rank 16 --lora_alpha 32.0 \
        --use_dynamic_lora --use_contrastive_loss \
        --use_multiscale_ot --use_ohem_loss --use_soct \
        --fp16
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.soct_sam_model import SOCT_SAM
import utils


@torch.no_grad()
def sweep_thresholds(model, loader, device, thresholds):
    """
    Run model inference ONCE, store raw sigmoid probabilities,
    then evaluate all thresholds without re-running the model.
    """
    model.eval()
    print("\n[Sweep] Running inference on val split (single pass)...")

    all_probs = []   # sigmoid probabilities per sample
    all_gt    = []   # ground truth masks per sample

    for batch_idx, (images, masks, captions) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        # Use first caption only (fast val protocol)
        if isinstance(captions[0], list):
            captions = [cap[0] for cap in captions]

        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images, captions, masks)

        probs = torch.sigmoid(outputs['pred_masks']).cpu()  # (B,1,H,W)
        gt    = masks.cpu().float()

        if probs.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, probs.shape[-2:], mode='nearest')

        for i in range(probs.shape[0]):
            all_probs.append(probs[i])
            all_gt.append(gt[i])

        if (batch_idx + 1) % 200 == 0:
            print(f"  Processed {batch_idx+1}/{len(loader)} batches...")

    print(f"[Sweep] Inference done — {len(all_probs)} samples collected.\n")

    # Evaluate each threshold
    results = []
    best_miou_row = None
    best_oiou_row = None

    print(f"{'Threshold':>12} | {'mIoU':>8} | {'oIoU':>8} | {'P@0.5':>8}")
    print("-" * 52)

    for thr in thresholds:
        ious        = []
        total_inter = 0.0
        total_union = 0.0
        p05_count   = 0

        for prob, gt in zip(all_probs, all_gt):
            pred_bin = (prob > thr).float()
            inter = (pred_bin * gt).sum().item()
            union = (pred_bin + gt - pred_bin * gt).sum().item()
            iou   = (inter + 1e-6) / (union + 1e-6)

            ious.append(iou)
            total_inter += inter
            total_union += union
            if iou > 0.5:
                p05_count += 1

        miou = float(np.mean(ious)) * 100.0
        oiou = (total_inter / (total_union + 1e-6)) * 100.0
        p05  = p05_count / max(len(ious), 1) * 100.0

        row = dict(threshold=thr, mIoU=miou, oIoU=oiou, P05=p05)
        results.append(row)

        print(f"  thr={thr:.3f}   | {miou:7.2f}% | {oiou:7.2f}% | {p05:7.2f}%")

        if best_miou_row is None or miou > best_miou_row['mIoU']:
            best_miou_row = row
        if best_oiou_row is None or oiou > best_oiou_row['oIoU']:
            best_oiou_row = row

    print("-" * 52)
    print(f"\n  Best mIoU  threshold: {best_miou_row['threshold']:.3f} "
          f"-> mIoU={best_miou_row['mIoU']:.2f}%  oIoU={best_miou_row['oIoU']:.2f}%")
    print(f"  Best oIoU  threshold: {best_oiou_row['threshold']:.3f} "
          f"-> mIoU={best_oiou_row['mIoU']:.2f}%  oIoU={best_oiou_row['oIoU']:.2f}%")

    return results, best_miou_row, best_oiou_row


def main():
    args = get_args()
    args.eval_only = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Threshold Sweep — SOCT-SAM")
    print(f"  Dataset : {args.dataset}  |  Split: {args.split}")
    print(f"  Resume  : {args.resume}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # Auto-resolve paths
    resolved_sam3 = utils.resolve_sam3_ckpt(args.sam3_ckpt)
    if resolved_sam3 != args.sam3_ckpt:
        args.sam3_ckpt = resolved_sam3

    resolved_data = utils.resolve_data_root(args.data_root, args.dataset)
    if resolved_data != args.data_root:
        args.data_root = resolved_data

    # Build Model
    print("Building model...")
    model = SOCT_SAM(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
        use_soct=getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False),
        lambda_occ=getattr(args, 'lambda_occ', 0.3),
        lambda_rho=getattr(args, 'lambda_rho', 0.2),
        lambda_tau=getattr(args, 'lambda_tau', 0.1),
        contrastive_weight=args.contrastive_weight,
        ohem_hard_ratio=args.ohem_hard_ratio,
        ot_reg=args.ot_reg,
        ot_num_iter=args.ot_num_iter,
        num_ot_scales=args.num_ot_scales,
        boundary_loss_weight=args.boundary_loss_weight,
        focal_gamma=args.focal_gamma,
    )

    # Load Checkpoint
    resume_path = args.resume
    if resume_path and not os.path.isfile(resume_path):
        for cand in [
            os.path.join(os.path.dirname(resume_path), 'best_model.pth'),
            os.path.join(args.output_dir, 'best_model.pth'),
        ]:
            if os.path.isfile(cand):
                resume_path = cand
                break

    if resume_path and os.path.isfile(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded epoch={ckpt.get('epoch','?')}  best_iou={ckpt.get('best_iou','?')}")
    else:
        print(f"WARNING: checkpoint not found at '{resume_path}'!")

    model = model.to(device).eval()

    # Dataset
    dataset = get_dataset(args, split=args.split, eval_mode=True)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Thresholds to sweep
    thresholds = [0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]

    # Run Sweep
    results, best_miou_row, best_oiou_row = sweep_thresholds(
        model, loader, device, thresholds
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f'threshold_sweep_{args.split}.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"Threshold Sweep on {args.dataset} ({args.split})\n")
        f.write(f"Checkpoint: {resume_path}\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Threshold':>10} | {'mIoU':>8} | {'oIoU':>8} | {'P@0.5':>8}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"  {r['threshold']:.3f}    | {r['mIoU']:7.2f}% | {r['oIoU']:7.2f}% | {r['P05']:7.2f}%\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best mIoU thr : {best_miou_row['threshold']:.3f}  "
                f"-> mIoU={best_miou_row['mIoU']:.2f}%  oIoU={best_miou_row['oIoU']:.2f}%\n")
        f.write(f"Best oIoU thr : {best_oiou_row['threshold']:.3f}  "
                f"-> mIoU={best_oiou_row['mIoU']:.2f}%  oIoU={best_oiou_row['oIoU']:.2f}%\n")

    print(f"\nResults saved to: {out_file}")
    print(f"\n{'='*60}")
    print(f"  NEXT STEP:")
    print(f"  Run test.py with --mask_threshold {best_oiou_row['threshold']:.2f}")
    print(f"  on --split test to see real test score.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
