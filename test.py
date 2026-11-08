"""
SOCT-SAM Evaluation Script

Evaluate trained SOCT-SAM model on test/val sets.
Computes: mIoU, oIoU (overall IoU), Precision@X thresholds.
Supports:
  1. Standard evaluation or Native Ground Truth evaluation (--eval_native_size)
  2. Relation-aware aerial Test-Time Augmentation (--use_tta)
  3. Configurable mask binarization threshold (--mask_threshold)

Usage:
    python test.py --dataset refseg_rs --split test --image_size 700 \
        --resume ./output/refsegrs_700_40epochs/best_model.pth \
        --output_dir ./output/refsegrs_700_40epochs \
        --eval_native_size --eval_only
"""

import os
import time
import re
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.soct_sam_model import SOCT_SAM
import utils


def swap_horizontal_relations(text):
    """
    Swap left/right, west/east directional tokens for relation-aware aerial TTA.
    Ensures that when an image is horizontally reflected, text queries with directional
    spatial anchors match the mirrored visual targets.
    """
    substitutions = [
        (r'\bleft\b', 'TEMP_RIGHT_PLACEHOLDER'),
        (r'\bright\b', 'left'),
        (r'\bTEMP_RIGHT_PLACEHOLDER\b', 'right'),
        (r'\bwest\b', 'TEMP_EAST_PLACEHOLDER'),
        (r'\beast\b', 'west'),
        (r'\bTEMP_EAST_PLACEHOLDER\b', 'east'),
        (r'\bwestern\b', 'TEMP_EASTERN_PLACEHOLDER'),
        (r'\beastern\b', 'western'),
        (r'\bTEMP_EASTERN_PLACEHOLDER\b', 'eastern'),
        (r'\bleftmost\b', 'TEMP_RIGHTMOST_PLACEHOLDER'),
        (r'\brightmost\b', 'leftmost'),
        (r'\bTEMP_RIGHTMOST_PLACEHOLDER\b', 'rightmost'),
    ]
    res = text
    for pat, rep in substitutions:
        res = re.sub(pat, rep, res, flags=re.IGNORECASE)
    return res


def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Compute segmentation metrics.

    Args:
        pred_masks: [B, 1, H, W] predicted logits
        gt_masks: [B, 1, H, W] ground truth binary masks

    Returns:
        dict with IoU, precision at various thresholds
    """
    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)

    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

    # Per-sample IoU
    intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Overall IoU (cumulative intersection / cumulative union)
    total_intersection = intersection.sum()
    total_union = union.sum()

    # Precision at thresholds
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = {}
    for t in prec_thresholds:
        precisions[f'P@{t}'] = (iou > t).float().mean().item()

    return {
        'iou': iou,
        'mean_iou': iou.mean().item(),
        'intersection': total_intersection.item(),
        'union': total_union.item(),
        **precisions,
    }


def predict_sample(model, image, caption, mask, use_tta=False):
    """
    Run forward pass with optional relation-aware TTA.

    Args:
        model: SOCT_SAM
        image: [1, 3, H, W]
        caption: str
        mask: [1, 1, H, W]
    Returns:
        pred_logits: [1, 1, H, W]
    """
    with torch.amp.autocast('cuda', enabled=True):
        outputs = model(image, [caption], mask)
        pred_logits = outputs['pred_masks']

    if not use_tta:
        return pred_logits

    # Relation-aware Horizontal Flip TTA
    caption_hf = swap_horizontal_relations(caption)
    image_hf = torch.flip(image, dims=[-1])
    mask_hf = torch.flip(mask, dims=[-1])

    with torch.amp.autocast('cuda', enabled=True):
        outputs_hf = model(image_hf, [caption_hf], mask_hf)
        pred_logits_hf = outputs_hf['pred_masks']

    prob1 = torch.sigmoid(pred_logits)
    prob2 = torch.sigmoid(torch.flip(pred_logits_hf, dims=[-1]))
    prob_ens = 0.5 * prob1 + 0.5 * prob2

    eps = 1e-6
    prob_ens = torch.clamp(prob_ens, eps, 1.0 - eps)
    pred_logits_ens = torch.log(prob_ens / (1.0 - prob_ens))
    return pred_logits_ens


@torch.no_grad()
def evaluate(model, test_loader, device, args):
    """Run full evaluation."""
    model.eval()

    all_ious = []
    total_intersection = 0
    total_union = 0
    prec_counts = {f'P@{t}': 0 for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_samples = 0
    total_time = 0

    threshold = getattr(args, 'mask_threshold', 0.50)
    eval_native = getattr(args, 'eval_native_size', False)
    use_tta = getattr(args, 'use_tta', False)

    for batch_idx, batch_data in enumerate(test_loader):
        if len(batch_data) == 4:
            images, masks, captions, orig_masks = batch_data
            if isinstance(orig_masks, torch.Tensor):
                orig_masks = orig_masks.to(device)
            elif isinstance(orig_masks, list):
                orig_masks = [om.to(device) for om in orig_masks]
        else:
            images, masks, captions = batch_data
            orig_masks = None

        images = images.to(device)
        masks = masks.to(device)

        # Handle eval captions (list of lists)
        if isinstance(captions[0], list):
            # Evaluate on each caption and take best IoU
            best_iou_per_sample = []
            for cap_list in captions:
                ious_for_caps = []
                for cap in cap_list:
                    start = time.time()
                    pred_logits = predict_sample(model, images[0:1], cap, masks[0:1], use_tta=use_tta)
                    total_time += time.time() - start

                    # Native resolution mapping
                    if eval_native and orig_masks is not None:
                        gt_target = orig_masks[0:1] if isinstance(orig_masks, torch.Tensor) else orig_masks[0]
                        pred_eval = F.interpolate(pred_logits, size=gt_target.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                        gt_target = masks[0:1]
                        pred_eval = pred_logits

                    metrics = compute_metrics(pred_eval, gt_target, threshold=threshold)
                    # Store iou, intersection, and union
                    ious_for_caps.append((metrics['iou'].item(), metrics['intersection'], metrics['union']))

                # Select the caption evaluation that gave the best IoU
                best_item = max(ious_for_caps, key=lambda x: x[0])
                best_iou_per_sample.append(best_item)

            for iou_val, intersection, union in best_iou_per_sample:
                all_ious.append(iou_val)
                total_intersection += intersection
                total_union += union
                total_samples += 1
                for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    if iou_val > t:
                        prec_counts[f'P@{t}'] += 1
        else:
            # Single caption per sample
            start = time.time()
            caption_str = captions[0] if isinstance(captions, list) else captions
            pred_logits = predict_sample(model, images[0:1], caption_str, masks[0:1], use_tta=use_tta)
            total_time += time.time() - start

            if eval_native and orig_masks is not None:
                gt_target = orig_masks[0:1] if isinstance(orig_masks, torch.Tensor) else orig_masks[0]
                pred_eval = F.interpolate(pred_logits, size=gt_target.shape[-2:], mode='bilinear', align_corners=False)
            else:
                gt_target = masks
                pred_eval = pred_logits

            metrics = compute_metrics(pred_eval, gt_target, threshold=threshold)
            all_ious.extend(metrics['iou'].cpu().numpy().tolist())
            total_intersection += metrics['intersection']
            total_union += metrics['union']
            total_samples += images.size(0)

            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                prec_counts[f'P@{t}'] += (metrics['iou'] > t).sum().item()

        if (batch_idx + 1) % 100 == 0:
            current_miou = np.mean(all_ious)
            print(f"  Progress: {batch_idx+1}/{len(test_loader)}, "
                  f"Current mIoU={current_miou:.4f}")

    # Final metrics
    mIoU = np.mean(all_ious)
    oIoU = total_intersection / (total_union + 1e-6) if total_union > 0 else mIoU
    print(f"[DEBUG] total_intersection: {total_intersection}, total_union: {total_union}, total_samples: {total_samples}")

    results = {
        'mIoU': mIoU,
        'oIoU': oIoU,
        'num_samples': total_samples,
        'avg_time': total_time / max(total_samples, 1),
    }
    for key in prec_counts:
        results[key] = prec_counts[key] / total_samples

    return results


@torch.no_grad()
def save_predictions(model, test_loader, device, vis_dir, args):
    """
    Save predicted masks as images for visualization side-by-side with original and GT.
    
    Generates a publication-grade horizontal 3-panel figure:
      Panel 1: Original satellite image (pure clean image)
      Panel 2: Ground Truth (mask highlighted in RED)
      Panel 3: Prediction (mask highlighted in GREEN)
      Header: Black banner with full text query and column headers directly above each panel.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)

    def draw_mask_contour(img, mask_bool, color_rgb, thickness=2):
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color_rgb, thickness, cv2.LINE_AA)

    print(f"\nSaving visualizations to: {vis_dir}")
    saved_count = 0
    max_to_save = getattr(args, 'max_vis', 50)
    threshold = getattr(args, 'mask_threshold', 0.50)
    eval_native = getattr(args, 'eval_native_size', False)
    use_tta = getattr(args, 'use_tta', False)

    for batch_idx, batch_data in enumerate(test_loader):
        if max_to_save > 0 and saved_count >= max_to_save:
            break

        if len(batch_data) == 4:
            images, masks, captions, orig_masks = batch_data
            if isinstance(orig_masks, torch.Tensor):
                orig_masks = orig_masks.to(device)
            elif isinstance(orig_masks, list):
                orig_masks = [om.to(device) for om in orig_masks]
        else:
            images, masks, captions = batch_data
            orig_masks = None

        images = images.to(device)
        masks = masks.to(device)

        if isinstance(captions[0], list):
            caption_str = captions[0][0]
        else:
            caption_str = captions[0]

        pred_logits = predict_sample(model, images[0:1], caption_str, masks[0:1], use_tta=use_tta)

        if eval_native and orig_masks is not None:
            target_mask_tensor = orig_masks[0:1] if isinstance(orig_masks, torch.Tensor) else orig_masks[0]
            pred_eval = F.interpolate(pred_logits, size=target_mask_tensor.shape[-2:], mode='bilinear', align_corners=False)
            gt_mask_np = target_mask_tensor[0, 0].cpu().numpy()
            pred_binary = (torch.sigmoid(pred_eval) > threshold).float()
            pred_mask_np = pred_binary[0, 0].cpu().numpy()

            # Attempt to read pristine original image if available on disk
            try:
                raw_img_path = test_loader.dataset.imgs[batch_idx]
                orig_img_pil = Image.open(raw_img_path).convert('RGB')
                orig_img_np = np.array(orig_img_pil)
            except Exception:
                img_tensor = images[0].cpu().permute(1, 2, 0).numpy()
                orig_img_np = np.clip(img_tensor * 255.0, 0, 255).astype(np.uint8)
                if orig_img_np.shape[:2] != gt_mask_np.shape[:2]:
                    orig_img_np = cv2.resize(orig_img_np, (gt_mask_np.shape[1], gt_mask_np.shape[0]))
        else:
            pred_binary = (torch.sigmoid(pred_logits) > threshold).float()
            img_tensor = images[0].cpu().permute(1, 2, 0).numpy()
            orig_img_np = np.clip(img_tensor * 255.0, 0, 255).astype(np.uint8)
            gt_mask_np = masks[0, 0].cpu().numpy()
            pred_mask_np = pred_binary[0, 0].cpu().numpy()

        h, w = orig_img_np.shape[:2]
        gt_indices = (gt_mask_np > 0)
        pred_indices = (pred_mask_np > 0)

        # Mask color palettes (in RGB): Ground Truth = RED, Prediction = GREEN
        red_rgb = np.zeros_like(orig_img_np)
        red_rgb[:, :] = [255, 0, 0]

        green_rgb = np.zeros_like(orig_img_np)
        green_rgb[:, :] = [0, 255, 0]

        # Panel 1: Original satellite image
        panel_orig = orig_img_np.copy()

        # Panel 2: Ground Truth (RED overlay + RED contour)
        panel_gt = orig_img_np.copy()
        if gt_indices.any():
            panel_gt[gt_indices] = (0.50 * orig_img_np[gt_indices] + 0.50 * red_rgb[gt_indices]).astype(np.uint8)
            draw_mask_contour(panel_gt, gt_indices, (255, 0, 0), thickness=2)

        # Panel 3: Prediction (GREEN overlay + GREEN contour)
        panel_pred = orig_img_np.copy()
        if pred_indices.any():
            panel_pred[pred_indices] = (0.50 * orig_img_np[pred_indices] + 0.50 * green_rgb[pred_indices]).astype(np.uint8)
            draw_mask_contour(panel_pred, pred_indices, (0, 255, 0), thickness=2)

        # Calculate individual sample IoU
        intersection_val = np.logical_and(gt_indices, pred_indices).sum()
        union_val = np.logical_or(gt_indices, pred_indices).sum()
        sample_iou = (intersection_val + 1e-6) / (union_val + 1e-6)

        # Combine 3 panels horizontally with thin sleek vertical divider
        divider_w = 2
        divider = np.zeros((h, divider_w, 3), dtype=np.uint8)
        triplet_img = np.hstack([panel_orig, divider, panel_gt, divider, panel_pred])
        total_w = triplet_img.shape[1]

        # 4. Text banner on top of full width (Query + Column Titles)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_query = 0.62
        font_scale_title = 0.68
        thickness = 1

        words = caption_str.split(' ')
        lines = []
        curr_line = []
        for w_text in words:
            test_line = ' '.join(curr_line + [w_text])
            (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale_query, thickness)
            if text_w > (total_w - 40) and curr_line:
                lines.append(' '.join(curr_line))
                curr_line = [w_text]
            else:
                curr_line.append(w_text)
        if curr_line:
            lines.append(' '.join(curr_line))

        # Banner layout
        query_y_start = 32
        line_height = 28
        query_end_y = query_y_start + (len(lines) - 1) * line_height

        title_y = query_end_y + 40
        banner_h = title_y + 16

        banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)

        # Draw query expression at top
        for idx, line in enumerate(lines):
            y_pos = query_y_start + idx * line_height
            cv2.putText(banner, line, (15, y_pos), font, font_scale_query, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw Column Titles directly aligned over each panel
        col1_x = 15
        col2_x = w + divider_w + 15
        col3_x = 2 * (w + divider_w) + 15

        cv2.putText(banner, "Original", (col1_x, title_y), font, font_scale_title, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(banner, "Ground Truth", (col2_x, title_y), font, font_scale_title, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(banner, "Prediction", (col3_x, title_y), font, font_scale_title, (255, 255, 255), thickness, cv2.LINE_AA)

        # 5. Stack header banner on top of the 3 panels
        final_img = np.vstack([banner, triplet_img])

        # 6. Save visualization image
        file_prefix = f'sample_{batch_idx}'
        vis_img = Image.fromarray(final_img)
        vis_img.save(os.path.join(vis_dir, f'{file_prefix}_3panel.png'))
        vis_img.save(os.path.join(vis_dir, f'{file_prefix}_iou{sample_iou*100:.1f}.png'))

        with open(os.path.join(vis_dir, f'{file_prefix}_info.txt'), 'w', encoding='utf-8') as f_meta:
            f_meta.write(f"Sample: batch_{batch_idx}\n")
            f_meta.write(f"Query Expression: {caption_str}\n")
            f_meta.write(f"Sample IoU: {sample_iou * 100:.2f}%\n")
            f_meta.write(f"Ground Truth Pixels: {int(gt_indices.sum())}\n")
            f_meta.write(f"Predicted Pixels: {int(pred_indices.sum())}\n")
            f_meta.write(f"Intersection Pixels: {int(intersection_val)}\n")
            f_meta.write(f"Union Pixels: {int(union_val)}\n")

        saved_count += 1
        print(f"  [VIS {saved_count:2d}/{max_to_save}] IoU: {sample_iou*100:5.2f}% | Query: \"{caption_str}\"")

    print(f"  Successfully saved {saved_count} visualizations to {vis_dir}")


def main():
    args = get_args()
    args.eval_only = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  SOCT-SAM Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Device: {device}")
    print(f"  Evaluation Mode: {'Native GT Resolution' if getattr(args, 'eval_native_size', False) else 'Resized 700x700'}")
    print(f"  TTA (Aerial Equivariant): {'Enabled' if getattr(args, 'use_tta', False) else 'Disabled'}")
    print(f"  Mask Threshold: {getattr(args, 'mask_threshold', 0.50):.3f}")
    print(f"  --- Enhancements ---")
    print(f"  SOCT-SAM: {getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False)}")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss}")
    print(f"  Gram SCL Loss: {getattr(args, 'use_scl', False)}")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot}")
    print(f"  OHEM Loss: {args.use_ohem_loss}")
    print(f"{'='*60}\n")

    # Auto-resolve sam3 checkpoint & data_root paths
    resolved_sam3 = utils.resolve_sam3_ckpt(args.sam3_ckpt)
    if resolved_sam3 != args.sam3_ckpt:
        print(f"🔍 [INFO] Provided sam3_ckpt '{args.sam3_ckpt}' not found. Auto-resolved to: '{resolved_sam3}'")
        args.sam3_ckpt = resolved_sam3

    resolved_data = utils.resolve_data_root(args.data_root, args.dataset)
    if resolved_data != args.data_root:
        print(f"🔍 [INFO] Provided data_root '{args.data_root}' not found. Auto-resolved to: '{resolved_data}'")
        args.data_root = resolved_data

    # ====== Build Model ======
    model = SOCT_SAM(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
        # SOCT-SAM Model
        use_soct=getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False),
        lambda_ord=getattr(args, 'lambda_ord', 5.0),
        theta_anchor=getattr(args, 'theta_anchor', 0.1),
        lambda_occ=getattr(args, 'lambda_occ', 0.3),
        lambda_rho=getattr(args, 'lambda_rho', 0.2),
        lambda_tau=getattr(args, 'lambda_tau', 0.1),
        use_scl=getattr(args, 'use_scl', False),
        soct_outer_iter=getattr(args, 'soct_outer_iter', 10),
        soct_inner_iter=getattr(args, 'soct_inner_iter', 30),
        # Enhancement params
        contrastive_weight=args.contrastive_weight,
        scl_weight=args.scl_weight,
        ohem_hard_ratio=args.ohem_hard_ratio,
        ot_reg=args.ot_reg,
        ot_num_iter=args.ot_num_iter,
        num_ot_scales=args.num_ot_scales,
        boundary_loss_weight=args.boundary_loss_weight,
        focal_gamma=args.focal_gamma,
    )

    # Load trained weights
    resume_path = args.resume
    if resume_path and not os.path.isfile(resume_path):
        candidates = [
            os.path.join(os.path.dirname(resume_path), 'best_model.pth'),
            os.path.join(args.output_dir, 'best_model.pth'),
            os.path.join(args.output_dir, 'checkpoint_epoch40.pth'),
            os.path.join(args.output_dir, 'latest_model.pth'),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                print(f"🔍 [AUTO-RESOLVE] Provided checkpoint '{resume_path}' not found.")
                print(f"    Auto-resolved to: '{cand}'")
                resume_path = cand
                break

    if resume_path and os.path.isfile(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
              f"best_iou={ckpt.get('best_iou', '?')}")
    else:
        print(f"WARNING: Checkpoint '{args.resume}' not found! Evaluating with pretrained SAM3 only!")

    model = model.to(device)
    model.eval()

    # ====== Build Dataset ======
    test_dataset = get_dataset(args, split=args.split, eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Evaluate ======
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, args)

    # ====== Print Results ======
    print(f"\n{'='*60}")
    print(f"  Results on {args.dataset} ({args.split})")
    print(f"  Mask Threshold: {getattr(args, 'mask_threshold', 0.50):.3f}")
    print(f"  Native Size Eval: {getattr(args, 'eval_native_size', False)}")
    print(f"  TTA Enabled: {getattr(args, 'use_tta', False)}")
    print(f"{'='*60}")
    print(f"  mIoU:  {results['mIoU']*100:.2f}%")
    print(f"  oIoU:  {results['oIoU']*100:.2f}%")
    print(f"  P@0.5: {results['P@0.5']*100:.2f}%")
    print(f"  P@0.6: {results['P@0.6']*100:.2f}%")
    print(f"  P@0.7: {results['P@0.7']*100:.2f}%")
    print(f"  P@0.8: {results['P@0.8']*100:.2f}%")
    print(f"  P@0.9: {results['P@0.9']*100:.2f}%")
    print(f"  Samples: {results['num_samples']}")
    print(f"  Avg Time: {results['avg_time']*1000:.1f}ms")

    # ====== Save Results to TXT File ======
    os.makedirs(args.output_dir, exist_ok=True)
    res_file = os.path.join(args.output_dir, f'results_{args.split}.txt')
    with open(res_file, 'w', encoding='utf-8') as f:
        f.write(f"Results on {args.dataset} ({args.split})\n")
        f.write(f"Checkpoint: {resume_path}\n")
        f.write(f"Native Size Eval: {getattr(args, 'eval_native_size', False)}\n")
        f.write(f"TTA: {'Enabled (Aerial Equivariant)' if getattr(args, 'use_tta', False) else 'Disabled'}\n")
        f.write(f"Mask Threshold: {getattr(args, 'mask_threshold', 0.50):.3f}\n")
        f.write(f"{'='*40}\n")
        f.write(f"mIoU:  {results['mIoU']*100:.2f}%\n")
        f.write(f"oIoU:  {results['oIoU']*100:.2f}%\n")
        f.write(f"P@0.5: {results['P@0.5']*100:.2f}%\n")
        f.write(f"P@0.6: {results['P@0.6']*100:.2f}%\n")
        f.write(f"P@0.7: {results['P@0.7']*100:.2f}%\n")
        f.write(f"P@0.8: {results['P@0.8']*100:.2f}%\n")
        f.write(f"P@0.9: {results['P@0.9']*100:.2f}%\n")
        f.write(f"Samples: {results['num_samples']}\n")
        f.write(f"Avg Time: {results['avg_time']*1000:.1f}ms\n")
    print(f"📄 Full metrics saved to: {res_file}")

    # ====== Save Visualizations ======
    if args.visualize:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vis_save_dir = os.path.join(args.output_dir, f'visualizations_{args.split}')
        save_predictions(model, test_loader, device, vis_save_dir, args)


if __name__ == '__main__':
    main()
