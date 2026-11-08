"""
SOCT-SAM Training Script

Train SOCT-SAM model for Referring Remote Sensing Image Segmentation:
    1. Stochastic-Order Constrained Transport (SOCT)
    2. Text-Guided Dynamic LoRA
    3. Multi-Scale Cost Lifting & Target Occupancy Prompting
    4. OHEM + Focal + Boundary Loss

Supports: RRSIS-D, RRSIS-HR, RefSegRS datasets.

Usage:
    python train.py --dataset rrsis_d --data_root /path/to/data --sam3_ckpt ./pre-trained-weights/sam3.pt
"""

import os
import sys
import time
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.soct_sam_model import SOCT_SAM
from lib.rs_adapters import get_trainable_params_summary
import utils

HAS_WANDB = False


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(pred, target, threshold=0.5, return_components=False):
    """Compute IoU between predicted and target masks."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    if return_components:
        return iou, intersection.sum().item(), union.sum().item()
    return iou.mean().item()


def get_optimizer(model, args):
    """
    Create optimizer with parameter groups for differentiated learning rates.

    Groups:
        1. LoRA / Dynamic LoRA parameters — lr_backbone
        2. Enhancement modules (OT, contrastive, OHEM) — lr_decoder
        3. Decoder/encoder parameters — lr_decoder
        4. Other trainable params — lr
    """
    lora_params = []
    enhancement_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower() or 'dynamic_lora' in name.lower() or 'hyper_' in name.lower():
            lora_params.append(param)
        elif any(x in name for x in ['soct_aligner', 'soct_loss', 'ms_ot_aligner', 'contrastive_loss', 'enhanced_loss', 'ot_aligner']):
            enhancement_params.append(param)
        elif any(x in name for x in ['transformer', 'segmentation_head', 'geometry_encoder', 'dot_prod']):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': lora_params, 'lr': args.lr_backbone, 'name': 'lora_adapters'},
        {'params': enhancement_params, 'lr': args.lr_decoder, 'name': 'enhancements'},
        {'params': decoder_params, 'lr': args.lr_decoder, 'name': 'decoder'},
        {'params': other_params, 'lr': args.lr, 'name': 'other'},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    for g in param_groups:
        print(f"  Param group '{g['name']}': {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler with warmup and minimum LR floor to prevent plateaus."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = getattr(args, 'warmup_epochs', 1) * steps_per_epoch
    min_lr_ratio = getattr(args, 'min_lr_ratio', 0.10)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        # Keep a 10% LR floor so late epochs never freeze into a plateau
        return (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


@torch.no_grad()
def validate(model, val_loader, device, epoch, args=None):
    """Run validation and compute metrics."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: '
    
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0
    total_its = 0

    for images, masks, captions in metric_logger.log_every(val_loader, 100, header):
        total_its += 1
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval mode captions (list of lists → flatten)
        if isinstance(captions[0], list):
            captions = [cap[0] for cap in captions]

        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images, captions, masks)

        loss = outputs['loss'].item()
        total_loss += loss

        iou_tensor, I, U = compute_iou(outputs['pred_masks'], masks, return_components=True)
        
        for iou in iou_tensor:
            iou_val = iou.item()
            mean_IoU.append(iou_val)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou_val >= eval_seg_iou)
            seg_total += 1

        cum_I += I
        cum_U += U

    mIoU = np.mean(mean_IoU)
    print('\n' + '='*58)
    print(f'  Epoch [{epoch}] Validation Results:')
    print('='*58)
    print('  mIoU:   %.2f%%' % (mIoU * 100.))
    overall_iou = (cum_I * 100. / cum_U) if cum_U > 0 else 0
    print('  oIoU:   %.2f%%' % overall_iou)
    for n_eval_iou in range(len(eval_seg_iou_list)):
        p_val = seg_correct[n_eval_iou] * 100. / max(seg_total, 1)
        print('  P@%s:  %.2f%%' % (str(eval_seg_iou_list[n_eval_iou]), p_val))
    print(f'  Samples Evaluated: {seg_total}')
    print('='*58 + '\n')

    # Append to permanent log file
    if args and getattr(args, 'output_dir', None):
        log_file = os.path.join(args.output_dir, 'val_metrics.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            p_str = ", ".join([f"P@{eval_seg_iou_list[k]}={seg_correct[k]*100./max(seg_total, 1):.2f}%" for k in range(len(eval_seg_iou_list))])
            f.write(f"Epoch {epoch:02d}: mIoU={mIoU*100.:.2f}%, oIoU={overall_iou:.2f}%, {p_str}\n")

    return mIoU * 100., overall_iou


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch with mixed precision, gradient accumulation, ETA, and loss logging."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f} ({avg:.4f})'))
    metric_logger.add_meter('iou', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    optimizer.zero_grad()

    for batch_idx, (images, masks, captions) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=args.fp16):
            outputs = model(images, captions, masks)
            loss = outputs['loss'] / args.grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Metrics
        with torch.no_grad():
            iou = compute_iou(outputs['pred_masks'], masks)

        # Log individual loss components
        log_dict = {
            'loss': loss.item() * args.grad_accum_steps,
            'iou': iou,
            'lr': optimizer.param_groups[0]["lr"],
        }
        if 'seg_loss' in outputs and isinstance(outputs['seg_loss'], torch.Tensor):
            log_dict['seg'] = outputs['seg_loss'].item()
        if 'contrastive_loss' in outputs and isinstance(outputs['contrastive_loss'], torch.Tensor):
            log_dict['cl'] = outputs['contrastive_loss'].item()
        if 'occ_loss' in outputs and isinstance(outputs['occ_loss'], torch.Tensor):
            log_dict['occ'] = outputs['occ_loss'].item()
        if 'area_loss' in outputs and isinstance(outputs['area_loss'], torch.Tensor):
            log_dict['area'] = outputs['area_loss'].item()

        metric_logger.update(**log_dict)
        del outputs, loss
        if (batch_idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return metric_logger.meters['loss'].global_avg, metric_logger.meters['iou'].global_avg


def main():
    args = get_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SOCT-SAM Training")
    print(f"  Dataset: {args.dataset}")
    print(f"  Image Size: {args.image_size}×{args.image_size}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size} × {args.grad_accum_steps} accum")
    print(f"  Device: {device}")
    print(f"  --- Enhancements ---")
    print(f"  SOCT-SAM: {getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False)} (lambda_ord={getattr(args, 'lambda_ord', 5.0)}, theta={getattr(args, 'theta_anchor', 0.1)})")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss} (weight={args.contrastive_weight})")
    print(f"  Gram SCL Loss: {getattr(args, 'use_scl', False)} (weight={args.scl_weight})")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot} ({args.num_ot_scales} scales)")
    print(f"  OHEM Loss: {args.use_ohem_loss} (hard_ratio={args.ohem_hard_ratio}, boundary_weight={args.boundary_loss_weight}, focal_gamma={args.focal_gamma})")
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
    print("Building SOCT-SAM model...")
    model = SOCT_SAM(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
        # Enhancement flags
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
    model = model.to(device)

    # Print rich Hyperparameters Table
    effective_bs = args.batch_size * args.grad_accum_steps
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if torch.cuda.is_available() else "N/A"

    print("\n" + "="*70)
    print("  SOCT-SAM Training Framework")
    print("="*70)
    print(f"  Dataset:              {args.dataset}")
    print(f"  Data Root:            {args.data_root}")
    print(f"  SAM3 Checkpoint:      {args.sam3_ckpt}")
    print(f"  Output Directory:     {args.output_dir}")
    print(f"  Device:               {device} ({gpu_name} - {gpu_mem})")
    print("  " + "-"*66)
    print(f"  Image Size:           {args.image_size} x {args.image_size}")
    print(f"  Batch Size (per GPU): {args.batch_size}")
    print(f"  Grad Accum Steps:     {args.grad_accum_steps} (Effective Batch Size: {effective_bs})")
    print(f"  Total Epochs:         {args.epochs} (Warmup Epochs: {getattr(args, 'warmup_epochs', 1)})")
    print(f"  Base Learning Rate:   {args.lr:.2e} (Min LR Floor: {args.lr * 0.1:.2e})")
    print(f"  Backbone (LoRA) LR:   {args.lr_backbone:.2e}")
    print(f"  Decoder LR:           {args.lr_decoder:.2e}")
    print(f"  Weight Decay:         {args.weight_decay}")
    print("  " + "-"*66)
    print(f"  LoRA Rank / Alpha:    {args.lora_rank} / {args.lora_alpha}")
    print(f"  Dynamic LoRA:         {args.use_dynamic_lora}")
    print(f"  SOCT Aligner:         {getattr(args, 'use_soct', True) and not getattr(args, 'no_soct', False)}")
    print(f"  Contrastive Loss:     {args.use_contrastive_loss} (weight: {args.contrastive_weight})")
    print(f"  OHEM Loss:            {args.use_ohem_loss} (ratio: {args.ohem_hard_ratio}, gamma: {args.focal_gamma})")
    print(f"  Multi-Scale OT:       {args.use_multiscale_ot} (scales: {args.num_ot_scales}, reg: {args.ot_reg})")
    print(f"  Mixed Precision:      {args.fp16}")
    print("="*70 + "\n")

    # ====== Build Datasets ======
    print("\nLoading datasets...")
    train_split = getattr(args, 'train_split', 'train')
    train_dataset = get_dataset(args, split=train_split, eval_mode=False)
    val_dataset = get_dataset(args, split='val', eval_mode=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Optimizer & Scheduler ======
    optimizer = get_optimizer(model, args)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    scheduler = get_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    # ====== Output Directory ======
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Resume / Auto-Resume ======
    start_epoch = 0
    best_iou = 0.0

    resume_target = args.resume
    if not resume_target:
        # Auto-detect existing checkpoint in output_dir
        cand_latest = os.path.join(args.output_dir, 'latest_model.pth')
        cand_best = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.isfile(cand_latest):
            print(f"🔍 [AUTO-RESUME] Found existing checkpoint '{cand_latest}'. Automatically resuming!")
            resume_target = cand_latest
        elif os.path.isfile(cand_best):
            print(f"🔍 [AUTO-RESUME] Found existing checkpoint '{cand_best}'. Automatically resuming!")
            resume_target = cand_best

    if resume_target:

        # 1. If path is a directory (e.g. unzipped PyTorch archive)
        if os.path.isdir(resume_target):
            found_dir = None
            if os.path.exists(os.path.join(resume_target, 'data.pkl')):
                found_dir = resume_target
            else:
                for root, _, files in os.walk(resume_target):
                    if 'data.pkl' in files:
                        found_dir = root
                        break

            if found_dir is not None:
                print(f"[Resume] Detected unzipped PyTorch checkpoint archive at: {found_dir}")
                reconstructed_pth = os.path.join(args.output_dir, 'reconstructed_resume.pth')
                print(f"[Resume] Reassembling archive into '{reconstructed_pth}'...")
                import zipfile
                parent_dir = os.path.dirname(found_dir)
                with zipfile.ZipFile(reconstructed_pth, 'w', compression=zipfile.ZIP_STORED) as z:
                    for root, _, files in os.walk(found_dir):
                        for f in files:
                            full_path = os.path.join(root, f)
                            rel_path = os.path.relpath(full_path, parent_dir)
                            z.write(full_path, rel_path)
                resume_target = reconstructed_pth
                print(f"[Resume] Checkpoint file successfully reconstructed!")

        # 2. If path doesn't directly exist, search common candidates
        if not os.path.isfile(resume_target):
            candidates = [
                resume_target,
                f"{resume_target}.pth",
                os.path.join(resume_target, 'best_model.pth'),
                os.path.join(resume_target, 'latest_model.pth'),
                os.path.join(args.output_dir, 'best_model.pth'),
                os.path.join(args.output_dir, 'latest_model.pth'),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resume_target = cand
                    break

        if not os.path.isfile(resume_target):
            raise FileNotFoundError(
                f"\n❌ [CRITICAL RESUME ERROR] --resume was provided ('{args.resume}'), but no valid checkpoint file (.pth) "
                f"or unzipped PyTorch archive directory was found!\n"
                f"Searched path: {resume_target}"
            )

        print(f"Resuming from {resume_target}")
        ckpt = torch.load(resume_target, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if getattr(args, 'finetune', False):
            print(f"  [Fine-Tune Mode] Successfully loaded weights from {resume_target}.")
            print(f"  Starting fresh fine-tune stage: 0 -> {args.epochs} epochs with fresh optimizer (lr={args.lr}).")
            start_epoch = 0
            best_iou = 0.0
        else:
            # Optionally load optimizer state, but update hyperparams to match new args
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] = args.weight_decay
                        if 'name' in param_group:
                            if param_group['name'] == 'lora_adapters':
                                param_group['lr'] = args.lr_backbone
                            elif param_group['name'] in ['enhancements', 'decoder']:
                                param_group['lr'] = args.lr_decoder
                            else:
                                param_group['lr'] = args.lr
                except Exception as e:
                    print(f"  [Warning] Could not load optimizer state (continuing with fresh optimizer): {e}")

            start_epoch = ckpt.get('epoch', 0)
            best_iou = ckpt.get('best_miou', None)
            if best_iou is None:
                best_iou = ckpt.get('best_iou', 0.0)
                if best_iou > 100.0:
                    # Legacy combined (mIoU + oIoU ~ 150.0). Scale to mIoU estimate
                    print(f"  [Auto-Fix] Legacy combined best_iou detected ({best_iou:.2f}%). Converting to standard mIoU scale.")
                    best_iou = best_iou / 2.0
                elif best_iou <= 1.0 and best_iou > 0.0:
                    # Fractional scale (e.g. 0.70 -> 70.0)
                    best_iou = best_iou * 100.0


            print(f"  Resumed at epoch {start_epoch}, best_mIoU={best_iou:.2f}%")
            
            # Fast-forward the scheduler to the correct step
            if start_epoch > 0:
                print(f"  Catching up scheduler to epoch {start_epoch} (fast-forwarding {start_epoch * steps_per_epoch} steps)...")
                for _ in range(start_epoch * steps_per_epoch):
                    scheduler.step()

    # ====== Wandb ======
    if HAS_WANDB:
        wandb.init(
            project="SOCT_SAM",
            name=f"{args.dataset}_soct_lr{args.lr}_lora{args.lora_rank}",
            config=vars(args),
        )

    # ====== Training Loop ======
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch + 1, args
        )

        # Validate according to val_freq or on the final epoch
        val_freq = getattr(args, 'val_freq', 1)
        run_val = ((epoch + 1) % val_freq == 0) or ((epoch + 1) == args.epochs)

        if run_val:
            val_iou, val_overall_iou = validate(model, val_loader, device, epoch + 1, args)
            print('Average object IoU {}'.format(val_iou))
            print('Overall IoU {}'.format(val_overall_iou))

            if HAS_WANDB:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_iou': train_iou,
                    'val_iou': val_iou,
                    'val_overall_iou': val_overall_iou,
                    'lr': optimizer.param_groups[0]['lr'],
                })

            # Save best model based on mIoU (standard primary benchmark metric in RRSIS literature)
            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou
                print('Better epoch: {}\n'.format(epoch + 1))
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                        'best_miou': val_iou,
                        'val_overall_iou': val_overall_iou,
                        'args': vars(args),
                    }, save_path)
                    print(f"  ★ New best model saved! (mIoU: {best_iou:.2f}%, oIoU: {val_overall_iou:.2f}%)")
                except Exception as e:
                    print(f"  ⚠️ [ERROR] Failed to save best_model.pth: {e}. Check available disk space!")
        else:
            print(f"  Skipping validation for epoch {epoch+1} (val_freq={val_freq}).")

        # Save latest model at every epoch with safe exception handling
        latest_save_path = os.path.join(args.output_dir, 'latest_model.pth')
        try:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args),
            }, latest_save_path)
            print(f"  Latest model weights updated for epoch {epoch+1}.")
        except Exception as e:
            print(f"  ⚠️ [ERROR] Failed to save latest_model.pth: {e}. Check available disk space!")

        # Save checkpoint every 10 epochs (Disabled by default to save disk space; only latest & best saved)
        if getattr(args, 'save_every_10', False) and (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'args': vars(args),
                }, save_path)
                print(f"  Checkpoint saved: {save_path}")
            except Exception as e:
                print(f"  ⚠️ [ERROR] Failed to save {save_path}: {e}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best mIoU: {best_iou:.4f}")
    print(f"{'='*60}")

    if HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
