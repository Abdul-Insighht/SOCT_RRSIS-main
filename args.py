import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation')

    # ====== Paths ======
    parser.add_argument('--data_root', type=str, default='./data/',
                        help='Root directory of datasets')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--sam3_ckpt', type=str, default='./pre-trained-weights/sam3.pt',
                        help='Path to SAM3 pretrained checkpoint')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume training from checkpoint')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Fine-tune mode: reset start_epoch to 0 and use fresh optimizer/scheduler for target resolution/epochs')

    # ====== Dataset ======
    parser.add_argument('--dataset', type=str, default='rrsis_d',
                        choices=[
                            'rrsis_d', 'rrsis-d', 'rrsisd',
                            'rrsis_hr', 'rrsis-hr', 'rrsishr',
                            'rislad', 'ris_lad', 'ris-lad',
                            'refsegrs', 'refseg-rs', 'refseg_rs'
                        ],
                        help='Dataset name: rrsis_d, rrsis_hr, ris-lad, refsegrs (hyphen & underscore aliases supported)')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Data split to train on (e.g. train, trainval)')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split (train/val/test)')
    parser.add_argument('--splitBy', type=str, default='unc',
                        help='Split method')
    parser.add_argument('--max_tokens', type=int, default=32,
                        help='Maximum text token length for SAM3 tokenizer')

    # ====== Model ======
    parser.add_argument('--image_size', type=int, default=800,
                        help='Input image size (divisible by 14 for SAM3 ViT)')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank for vision backbone adaptation')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout rate to prevent overfitting on foundation model')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze SAM3 ViT backbone (train only LoRA + decoder)')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True,
                        help='Freeze SAM3 text encoder')

    # ====== Enhancement Flags (NEW) ======
    parser.add_argument('--use_dynamic_lora', action='store_true', default=True,
                        help='Enable text-guided dynamic LoRA adapters')
    parser.add_argument('--no_dynamic_lora', action='store_true', default=False,
                        help='Disable dynamic LoRA (use static LoRA instead)')
    parser.add_argument('--use_contrastive_loss', action='store_true', default=True,
                        help='Enable contrastive language-image loss (InfoNCE)')
    parser.add_argument('--no_contrastive_loss', action='store_true', default=False,
                        help='Disable contrastive loss')
    parser.add_argument('--use_multiscale_ot', action='store_true', default=True,
                        help='Enable multi-scale OT feature alignment')
    parser.add_argument('--no_multiscale_ot', action='store_true', default=False,
                        help='Disable multi-scale OT (use single-scale)')
    parser.add_argument('--use_ohem_loss', action='store_true', default=True,
                        help='Enable OHEM + Focal + Boundary loss')
    parser.add_argument('--no_ohem_loss', action='store_true', default=False,
                        help='Disable OHEM loss (use standard Dice+BCE)')

    # ====== SOCT-SAM Flags ======
    parser.add_argument('--use_soct', action='store_true', default=True,
                        help='Enable Stochastic-Order Constrained Transport (SOCT)')
    parser.add_argument('--no_soct', action='store_true', default=False,
                        help='Disable SOCT (fallback to baseline ROTSAM)')
    parser.add_argument('--lambda_ord', type=float, default=5.0,
                        help='SOCT: Stochastic order penalty multiplier')
    parser.add_argument('--theta_anchor', type=float, default=0.1,
                        help='SOCT: Contextual anchor belief temperature')
    parser.add_argument('--lambda_occ', type=float, default=0.3,
                        help='SOCT: Weight for target occupancy loss (L_occ)')
    parser.add_argument('--lambda_rho', type=float, default=0.2,
                        help='SOCT: Weight for target area Huber loss (L_rho)')
    parser.add_argument('--lambda_tau', type=float, default=0.1,
                        help='SOCT: Weight for adaptive bandwidth regularization (L_tau)')
    parser.add_argument('--use_scl', action='store_true', default=False,
                        help='Enable Gram matrix SCL loss (ablation C4 only; removed from full model)')
    parser.add_argument('--soct_outer_iter', type=int, default=10,
                        help='SOCT: Dual projected gradient ascent iterations')
    parser.add_argument('--soct_inner_iter', type=int, default=30,
                        help='SOCT: Inner log-domain Sinkhorn iterations per dual update')

    # ====== Enhancement Parameters (NEW) ======
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss (InfoNCE)')
    parser.add_argument('--scl_weight', type=float, default=0.1,
                        help='Weight for structural consistency loss (SCL)')
    parser.add_argument('--ohem_hard_ratio', type=float, default=0.3,
                        help='OHEM: fraction of hardest pixels to keep')
    parser.add_argument('--ot_reg', type=float, default=0.1,
                        help='Sinkhorn OT entropy regularization')
    parser.add_argument('--ot_num_iter', type=int, default=10,
                        help='Number of Sinkhorn iterations')
    parser.add_argument('--num_ot_scales', type=int, default=3,
                        help='Number of FPN scales for multi-scale OT')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.5,
                        help='Weight for boundary-aware loss component')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (higher = more focus on hard)')

    # ====== Training ======
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Base learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for backbone (LoRA params)')
    parser.add_argument('--lr_decoder', type=float, default=5e-5,
                        help='Learning rate for decoder/seg head')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr_ratio', type=float, default=0.15,
                        help='Minimum LR floor ratio for cosine schedule to prevent late-epoch plateau/freezing')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')

    # ====== Loss ======
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--ce_weight', type=float, default=0.5,
                        help='Weight for Cross-Entropy loss')
    parser.add_argument('--boundary_weight', type=float, default=0.2,
                        help='Weight for boundary loss')

    # ====== Optimization ======
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision (fp16/bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing for memory savings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency')

    # ====== Distributed ======
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs')

    # ====== Evaluation ======
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Run evaluation only')
    parser.add_argument('--eval_native_size', action='store_true', default=False,
                        help='Evaluate prediction interpolated to native/original ground truth mask resolution')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Enable test-time augmentation (relation-aware aerial horizontal flip)')
    parser.add_argument('--mask_threshold', type=float, default=0.50,
                        help='Sigmoid threshold for mask binarization (default: 0.50). '
                             'Use threshold_sweep.py to find the optimal value per checkpoint.')
    parser.add_argument('--visualize', '--save_vis', action='store_true', default=False,
                        dest='visualize', help='Save visualization of predictions')
    parser.add_argument('--max_vis', type=int, default=50,
                        help='Maximum number of qualitative visualization samples to save (-1 for all)')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency in epochs (default: 1, e.g. set 5 to validate every 5 epochs)')

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    # Handle --no_* flags to override defaults
    if args.no_dynamic_lora:
        args.use_dynamic_lora = False
    if args.no_contrastive_loss:
        args.use_contrastive_loss = False
    if args.no_multiscale_ot:
        args.use_multiscale_ot = False
    if args.no_ohem_loss:
        args.use_ohem_loss = False

    return args
