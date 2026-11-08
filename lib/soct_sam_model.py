"""
SOCT_SAM: Main model architecture for Referring Remote Sensing Image Segmentation.

Key Components:
    1. Stochastic-Order Constrained Transport (SOCT) Alignment
    2. Text-Guided Dynamic LoRA
    3. Multi-Scale Feature Alignment
    4. Grounding-Aware Target Occupancy Point Prompting
    5. OHEM + Focal + Boundary Loss

Architecture:
    Input (Image + Text)
        -> SAM3 VL Backbone (ViT + Text Encoder)
            + Dynamic LoRA (text-conditioned vision adaptation)
        -> SOCT Feature Alignment & Target Occupancy Prompting
        -> Transformer Encoder (cross-modal fusion)
        -> DETR Decoder (detection & query scoring)
        -> Segmentation Head (mask prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sam3.model_builder import (
    build_sam3_image_model,
    _create_text_encoder,
    _create_vision_backbone,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
    _load_checkpoint,
)
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt

from .rs_adapters import inject_lora_adapters, get_trainable_params_summary
from .dynamic_lora import inject_dynamic_lora_adapters, DynamicLoRAManager
from .multiscale_ot_alignment import MultiScaleOTAligner
from .contrastive_loss import ContrastiveLoss
from .ohem_loss import EnhancedOHEMLoss
from .ot_feature_alignment import OTFeatureAligner
from .ot_loss import OTSegmentationLoss
from .prompt_generator import GroundingAwarePromptGenerator
from .soct_alignment import SOCTAligner
from .soct_loss import SOCTSupervisionLoss
from .language_parser import GLOBAL_PARSER


class SOCT_SAM(nn.Module):
    """
    SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation.

    Extends SAM 3 foundation model with:
    - Stochastic-Order Constrained Transport (SOCT) with directional ordering constraints
    - Target occupancy geometric point prompting (GPG)
    - Text-Guided Dynamic LoRA adapters
    - Multi-scale visual-linguistic feature alignment
    - OHEM + Focal + Boundary loss supervision

    All enhancements are individually toggleable for ablation.

    Args:
        sam3_ckpt: Path to SAM3 pretrained checkpoint.
        image_size: Input image size (divisible by 14).
        lora_rank: LoRA rank for adapters.
        lora_alpha: LoRA scaling factor.
        freeze_backbone: Whether to freeze the ViT backbone.
        freeze_text_encoder: Whether to freeze the text encoder.
        gradient_checkpointing: Enable gradient checkpointing.
        use_dynamic_lora: Enable text-guided dynamic LoRA.
        use_contrastive_loss: Enable contrastive loss.
        use_multiscale_ot: Enable multi-scale OT alignment.
        use_ohem_loss: Enable OHEM + Focal + Boundary loss.
        contrastive_weight: Weight for contrastive loss.
        ohem_hard_ratio: OHEM hard pixel ratio.
        ot_reg: OT Sinkhorn regularization.
        ot_num_iter: Number of Sinkhorn iterations.
        num_ot_scales: Number of FPN scales for multi-scale OT.
    """

    def __init__(
        self,
        sam3_ckpt: str = None,
        image_size: int = 504,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        freeze_backbone: bool = True,
        freeze_text_encoder: bool = True,
        gradient_checkpointing: bool = True,
        # === Enhancement flags ===
        use_dynamic_lora: bool = True,
        use_contrastive_loss: bool = True,
        use_multiscale_ot: bool = True,
        use_ohem_loss: bool = True,
        # === SOCT-SAM Model ===
        use_soct: bool = True,
        lambda_ord: float = 5.0,
        theta_anchor: float = 0.1,
        lambda_occ: float = 0.3,
        lambda_rho: float = 0.2,
        lambda_tau: float = 0.1,
        use_scl: bool = False,
        soct_outer_iter: int = 10,
        soct_inner_iter: int = 30,
        # === Enhancement params ===
        contrastive_weight: float = 0.1,
        scl_weight: float = 0.1,
        ohem_hard_ratio: float = 0.3,
        ot_reg: float = 0.1,
        ot_num_iter: int = 10,
        num_ot_scales: int = 3,
        boundary_loss_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_dynamic_lora = use_dynamic_lora
        self.use_contrastive_loss = use_contrastive_loss
        self.use_multiscale_ot = use_multiscale_ot
        self.use_ohem_loss = use_ohem_loss
        self.use_soct = use_soct
        self.lambda_ord = lambda_ord
        self.theta_anchor = theta_anchor
        self.lambda_occ = lambda_occ
        self.lambda_rho = lambda_rho
        self.lambda_tau = lambda_tau
        self.use_scl = use_scl
        self.contrastive_weight = contrastive_weight
        self.scl_weight = scl_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.focal_gamma = focal_gamma

        # ====== Build SAM3 Image Model ======
        print("[SOCT_SAM] Building SAM3 image model...")
        self.sam3 = build_sam3_image_model(
            device="cpu",
            eval_mode=False,
            checkpoint_path=sam3_ckpt,
            load_from_HF=(sam3_ckpt is None),
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

        # ====== Freeze Strategy ======
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_text_encoder:
            self._freeze_text_encoder()

        # ====== LoRA Injection ======
        d_model = self.sam3.hidden_dim  # typically 256

        if use_dynamic_lora:
            print("[SOCT_SAM] Injecting Text-Guided Dynamic LoRA...")
            _, self.lora_manager = inject_dynamic_lora_adapters(
                self.sam3, text_dim=d_model, rank=lora_rank, alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            print("[SOCT_SAM] Using standard static LoRA...")
            inject_lora_adapters(self.sam3, rank=lora_rank, alpha=lora_alpha, lora_dropout=lora_dropout)
            self.lora_manager = None

        # ====== Unfreeze trainable components ======
        self._unfreeze_trainable_components()

        # ====== Gradient Checkpointing ======
        if gradient_checkpointing:
            print("[SOCT_SAM] Gradient checkpointing enabled")

        # ====== Alignment Module ======
        if use_soct:
            print(f"[SOCT_SAM] SOCT-SAM Alignment enabled (lambda_ord={lambda_ord}, theta={theta_anchor})")
            self.soct_aligner = SOCTAligner(
                d_model=d_model,
                num_scales=num_ot_scales,
                lambda_ord=lambda_ord,
                theta_anchor=theta_anchor,
                ot_eps=ot_reg,
                num_outer=soct_outer_iter,
                num_inner=soct_inner_iter,
            )
            self.soct_loss = SOCTSupervisionLoss(
                lambda_occ=lambda_occ,
                lambda_rho=lambda_rho,
                lambda_tau=lambda_tau,
            )
        elif use_multiscale_ot:
            print(f"[SOCT_SAM] Multi-Scale OT Alignment ({num_ot_scales} scales) [Baseline]")
            self.ms_ot_aligner = MultiScaleOTAligner(
                d_model=d_model,
                num_scales=num_ot_scales,
                reg=ot_reg,
                num_iter=ot_num_iter,
            )
        else:
            # Fallback to single-scale OT from RRSIS_SAM3
            print("[SOCT_SAM] Single-Scale OT Alignment (baseline)")
            self.ot_aligner = OTFeatureAligner(
                d_model=d_model,
                reg=ot_reg,
                num_iter=ot_num_iter,
            )

        # ====== Contrastive Loss ======
        if use_contrastive_loss:
            print("[SOCT_SAM] Contrastive Loss (InfoNCE) enabled")
            self.contrastive_loss = ContrastiveLoss(
                visual_dim=d_model,
                text_dim=d_model,
            )

        # ====== OHEM Loss ======
        if use_ohem_loss:
            print("[SOCT_SAM] OHEM + FocalDice + Boundary Loss enabled")
            self.enhanced_loss = EnhancedOHEMLoss(
                hard_ratio=ohem_hard_ratio,
                boundary_weight=boundary_loss_weight,
                focal_gamma=focal_gamma
            )
        else:
            print("[SOCT_SAM] Standard Dice+BCE Loss (baseline)")
            self.standard_loss = OTSegmentationLoss()

        # ====== Grounding-Aware Prompt Generator ======
        print("[SOCT_SAM] Grounding-Aware Prompt Generation (GPG) enabled")
        self.gpg = GroundingAwarePromptGenerator(max_points=5, min_separation=2)

        # ====== Print Summary ======
        get_trainable_params_summary(self)

        # Image normalization
        self.register_buffer('pixel_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    # ================================================================
    # Freeze / Unfreeze helpers
    # ================================================================

    def _freeze_backbone(self):
        """Freeze the ViT vision backbone."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'vision_backbone'):
            for param in backbone.vision_backbone.parameters():
                param.requires_grad = False
            print("[SOCT_SAM] ViT backbone frozen")

    def _freeze_text_encoder(self):
        """Freeze the text encoder."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'language_backbone'):
            for param in backbone.language_backbone.parameters():
                param.requires_grad = False
            print("[SOCT_SAM] Text encoder frozen")

    def _unfreeze_trainable_components(self):
        """Unfreeze components we want to fine-tune."""
        if hasattr(self.sam3, 'transformer'):
            for param in self.sam3.transformer.encoder.parameters():
                param.requires_grad = True
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True
            print("[SOCT_SAM] Transformer encoder+decoder unfrozen")

        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True
            print("[SOCT_SAM] Segmentation head unfrozen")

        if hasattr(self.sam3, 'geometry_encoder'):
            for param in self.sam3.geometry_encoder.parameters():
                param.requires_grad = True
            print("[SOCT_SAM] Geometry encoder unfrozen")

        if hasattr(self.sam3, 'dot_prod_scoring') and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
            print("[SOCT_SAM] Scoring head unfrozen")

    def normalize_image(self, images):
        """Normalize images to SAM3's expected range [-1, 1]."""
        return (images - self.pixel_mean) / self.pixel_std

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, images, captions, masks_gt=None):
        """
        Enhanced forward pass with all 4 techniques.

        Args:
            images: [B, 3, H, W] tensor of RS images.
            captions: List[str] of length B, referring text descriptions.
            masks_gt: [B, 1, H, W] ground truth masks (for loss).

        Returns:
            dict with 'pred_masks', 'pred_logits', 'pred_boxes', 'loss'.
        """
        B = images.shape[0]
        device = images.device

        # Normalize images
        images = self.normalize_image(images)

        # ====== Step 1: Forward Text First ======
        text_out = self.sam3.backbone.forward_text(captions, device=device)

        # ====== Step 1.5: Dynamic LoRA text conditioning ======
        if self.use_dynamic_lora and self.lora_manager is not None:
            text_feats = text_out.get('language_features', None)
            if text_feats is not None:
                # Pool text features for conditioning: (seq, B, C) â†’ (B, C)
                if text_feats.dim() == 3:
                    pooled_text = text_feats.mean(dim=0)  # (B, C)
                else:
                    pooled_text = text_feats
                self.lora_manager.set_text_conditioning(pooled_text)

        # ====== Step 1.8: Forward Image (Now with Text Context!) ======
        backbone_out = self.sam3.backbone.forward_image(images)
        backbone_out.update(text_out)

        # ====== Step 2: Alignment (SOCT or Multi-Scale OT) ======
        text_feats = backbone_out.get('language_features', None)
        text_mask = backbone_out.get('language_mask', None)

        q_finest = None
        rho_hat = None
        tau_val = None
        P_map = None

        if text_feats is not None and 'backbone_fpn' in backbone_out:
            fpn_feats = backbone_out['backbone_fpn']

            if self.use_soct and hasattr(self, 'soct_aligner'):
                # SOCT-SAM: Parse language roles and run Stochastic-Order Constrained Transport
                parsed_batch = []
                for cap in captions:
                    p_info = GLOBAL_PARSER.parse(cap)
                    words = cap.lower().split()
                    t_idx, a_rels = GLOBAL_PARSER.align_tokens_to_roles(words, p_info)
                    parsed_batch.append({
                        'target_indices': t_idx,
                        'relations': a_rels,
                        'info': p_info
                    })

                aligned_fpn, q_finest, rho_hat, tau_val = self.soct_aligner(
                    fpn_features=fpn_feats,
                    text_feats=text_feats,
                    text_mask=text_mask,
                    parsed_batch=parsed_batch,
                )
                backbone_out['backbone_fpn'] = aligned_fpn

            elif self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
                # Multi-Scale OT alignment (Baseline B0)
                aligned_fpn = self.ms_ot_aligner(fpn_feats, text_feats, text_mask)
                backbone_out['backbone_fpn'] = aligned_fpn
                last_aligner = self.ms_ot_aligner.scale_aligners[-1]
                P_map = getattr(last_aligner, 'last_P', None)
            elif hasattr(self, 'ot_aligner'):
                # Single-scale OT alignment (baseline fallback)
                for i in range(len(fpn_feats)):
                    feat = fpn_feats[i]
                    if hasattr(feat, 'tensors'):
                        feat_tensor = feat.tensors
                    else:
                        feat_tensor = feat
                    if feat_tensor.dim() == 4:
                        aligned = self.ot_aligner(feat_tensor, text_feats, text_mask)
                        if hasattr(feat, 'tensors'):
                            feat.tensors = aligned
                        else:
                            fpn_feats[i] = aligned
                P_map = getattr(self.ot_aligner, 'last_P', None)

        # ====== Step 3: Prompt Generation via GPG Module ======
        img_ids = torch.arange(B, device=device)
        text_ids = torch.arange(B, device=device)

        input_points = torch.zeros(B, 0, 2, device=device)
        input_points_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
        input_point_labels = torch.zeros(B, 0, device=device, dtype=torch.long)
        has_prompts = False

        if self.use_soct and q_finest is not None:
            # Generate explicit geometric prompts from finest-grid target occupancy q^(f)
            input_points, input_points_mask, input_point_labels = self.gpg(
                occupancy=q_finest,
                original_image_size=self.image_size,
                device=device,
                rho_hat=rho_hat,
            )
            has_prompts = True
        elif P_map is not None:
            # Baseline fallback from transport plan
            input_points, input_points_mask, input_point_labels = self.gpg(
                occupancy=P_map,
                original_image_size=self.image_size,
                device=device,
                text_mask=text_mask,
            )
            has_prompts = True

        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros(B, 0, 4, device=device),
            input_boxes_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            input_boxes_label=torch.zeros(B, 0, device=device, dtype=torch.long),
            input_points=input_points,
            input_points_mask=input_points_mask,
        )

        if has_prompts:
            # point_embeddings: [N_points, B, 2] - Must be normalized to [0, 1] for SAM3 pos encoder
            point_embeddings = (input_points / self.image_size).transpose(0, 1)
            # point_labels: [N_points, B]
            point_labels = input_point_labels.transpose(0, 1)

            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, B, 4, device=device),
                box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
                point_embeddings=point_embeddings,
                point_mask=input_points_mask,
                point_labels=point_labels,
            )
        else:
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, B, 4, device=device),
                box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            )

        # ====== Step 4: Encode Prompt ======
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )

        # ====== Step 5: Run Encoder (fusion) ======
        backbone_out, encoder_out, feat_tuple = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        # ====== Step 6: Run Decoder (DETR detection) ======
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        }
        out, hs = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # ====== Step 7: Segmentation Head ======
        if self.sam3.segmentation_head is not None:
            _, _, _, vis_feat_sizes = feat_tuple
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]

            self.sam3._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=vis_feat_sizes,
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        # ====== Step 8: Select Best Mask ======
        result = self._select_best_mask(
            out, B,
            masks_gt=masks_gt if self.training else None,
            q_finest=q_finest,
        )

        # ====== Step 9: Clear Dynamic LoRA conditioning ======
        # if self.use_dynamic_lora and self.lora_manager is not None:
        #     self.lora_manager.clear_text_conditioning()

        # ====== Step 10: Compute Losses ======
        if masks_gt is not None:
            # Primary segmentation loss
            if self.use_ohem_loss and hasattr(self, 'enhanced_loss'):
                seg_loss = self.enhanced_loss(result, masks_gt, self.image_size)
            elif hasattr(self, 'standard_loss'):
                seg_loss = self.standard_loss(result, masks_gt, self.image_size)
            else:
                seg_loss = self._compute_fallback_loss(result['pred_masks'], masks_gt)

            # SOCT Auxiliary Losses (Occupancy Loss, Area Loss, Bandwidth Loss)
            soct_aux_loss = torch.tensor(0.0, device=device)
            if self.use_soct and hasattr(self, 'soct_loss') and q_finest is not None and rho_hat is not None:
                H_f, W_f = q_finest.shape[-2:]
                tau_0 = 1.0 / max(H_f, W_f)
                soct_aux_loss, l_occ, l_rho, l_tau = self.soct_loss(
                    q_finest=q_finest,
                    rho_hat=rho_hat,
                    gt_masks=masks_gt,
                    tau_val=tau_val,
                    tau_0=tau_0,
                )
                result['occ_loss'] = l_occ.detach()
                result['area_loss'] = l_rho.detach()
                result['tau_loss'] = l_tau.detach()
                result['pred_rho'] = rho_hat.detach().mean()

            # SCL Loss (Structural Consistency Loss - Ablation C4 only)
            scl_loss = torch.tensor(0.0, device=device)
            if self.use_scl and hasattr(self, 'ms_ot_aligner'):
                scl_vals = [a.scl_loss for a in self.ms_ot_aligner.scale_aligners if isinstance(getattr(a, 'scl_loss', 0.0), torch.Tensor)]
                if scl_vals:
                    scl_loss = sum(scl_vals) / len(scl_vals)

            # Contrastive loss (auxiliary)
            contrastive = torch.tensor(0.0, device=device)
            if self.use_contrastive_loss and hasattr(self, 'contrastive_loss'):
                try:
                    # Use FPN features with real spatial structure (not pooled encoder output)
                    fpn_feats = backbone_out.get('backbone_fpn', None)
                    if fpn_feats is not None and text_feats is not None and len(fpn_feats) > 0:
                        # Use highest-resolution FPN level for spatial contrastive learning
                        vis_feat = fpn_feats[0]
                        if hasattr(vis_feat, 'tensors'):
                            vis_feat = vis_feat.tensors
                        if vis_feat.dim() == 4:  # (B, C, H, W)
                            contrastive = self.contrastive_loss(
                                vis_feat, text_feats,
                                result['pred_masks'], masks_gt
                            )
                except Exception as e:
                    # Don't crash training if contrastive loss fails
                    if self.training:
                        print(f"[WARNING] Contrastive loss skipped: {e}")
                    contrastive = torch.tensor(0.0, device=device)

            total_loss = seg_loss + self.contrastive_weight * contrastive + soct_aux_loss
            if self.use_scl:
                total_loss = total_loss + self.scl_weight * scl_loss

            result['loss'] = total_loss
            result['seg_loss'] = seg_loss.detach()
            result['contrastive_loss'] = contrastive.detach() if isinstance(contrastive, torch.Tensor) else contrastive
            if self.use_scl:
                result['scl_loss'] = scl_loss.detach() if isinstance(scl_loss, torch.Tensor) else scl_loss

        return result

    def _select_best_mask(self, out, batch_size, masks_gt=None, q_finest=None):
        """Select the best mask from SAM3's multi-query output."""
        result = {}

        if q_finest is not None:
            result['q_finest'] = q_finest

        semantic_seg = out.get('semantic_seg', None)
        if semantic_seg is not None:
            result['semantic_seg'] = semantic_seg

        pred_logits = out.get('pred_logits', None)
        if pred_logits is not None:
            result['pred_logits'] = pred_logits

        pred_boxes = out.get('pred_boxes', None)
        if pred_boxes is not None:
            result['pred_boxes'] = pred_boxes

        pred_masks = out.get('pred_masks', None)
        if pred_masks is not None:
            # Store all query masks for IoU-based score supervision
            result['all_query_masks'] = pred_masks  # (B, N, H_mask, W_mask)

            if pred_logits is not None:
                # During training with GT available: use IoU matching to select the winning query
                # so the best query hypothesis receives mask gradients and the classifier
                # learns to pick it. During eval/inference, use SOCT-guided consensus + classifier.
                if self.training and masks_gt is not None:
                    with torch.no_grad():
                        gt_for_iou = masks_gt.float()
                        if pred_masks.shape[-2:] != gt_for_iou.shape[-2:]:
                            gt_for_iou = F.interpolate(gt_for_iou, size=pred_masks.shape[-2:], mode='nearest')
                        gt_exp = gt_for_iou.squeeze(1).unsqueeze(1).expand_as(pred_masks)
                        p_bin = (torch.sigmoid(pred_masks) > 0.5).float()
                        inter = (p_bin * gt_exp).sum(dim=(2, 3))
                        union = p_bin.sum(dim=(2, 3)) + gt_exp.sum(dim=(2, 3)) - inter
                        query_ious = (inter + 1e-6) / (union + 1e-6)

                        has_overlap = (inter > 0).any(dim=-1)
                        best_iou_idx = query_ious.argmax(dim=-1)
                        scores = pred_logits.squeeze(-1)
                        best_score_idx = scores.argmax(dim=-1)

                        best_idx = torch.where(has_overlap, best_iou_idx, best_score_idx)
                    batch_idx = torch.arange(batch_size, device=pred_masks.device)
                    best_masks = pred_masks[batch_idx, best_idx].unsqueeze(1)
                else:
                    scores = pred_logits.squeeze(-1).float()  # (B, N)
                    soct_w = getattr(self, 'soct_query_weight', 0.35)
                    topk_q = getattr(self, 'eval_topk_queries', 1)

                    if q_finest is not None and soct_w > 0.0:
                        with torch.no_grad():
                            q_ref = F.interpolate(
                                q_finest.unsqueeze(1).float(),
                                size=pred_masks.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )  # (B, 1, H_m, W_m)
                            # Normalize q_ref to [0, 1] per sample so sparse transport mass scales cleanly
                            q_max = q_ref.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                            q_norm = (q_ref / q_max).clamp(0.0, 1.0)

                            p_prob = torch.sigmoid(pred_masks.float())  # (B, N, H_m, W_m)
                            inter_q = (p_prob * q_norm).sum(dim=(2, 3))
                            union_q = p_prob.sum(dim=(2, 3)) + q_norm.sum(dim=(2, 3)) - inter_q
                            soct_iou = (inter_q + 1e-6) / (union_q + 1e-6)  # (B, N)

                            # Normalize classifier probabilities across Top-10 candidates to prevent low-rank noise
                            cls_prob = torch.sigmoid(scores)
                            cls_norm = cls_prob / (cls_prob.max(dim=-1, keepdim=True)[0] + 1e-6)
                            soct_norm = soct_iou / (soct_iou.max(dim=-1, keepdim=True)[0] + 1e-6)

                            # Restrict re-ranking to top-10 classifier queries
                            top_m = min(10, scores.shape[-1])
                            _, topm_indices = torch.topk(scores, k=top_m, dim=-1)
                            topm_mask = torch.zeros_like(scores, dtype=torch.bool)
                            topm_mask.scatter_(1, topm_indices, True)

                            combined_scores = (1.0 - soct_w) * cls_norm + soct_w * soct_norm
                            combined_scores = combined_scores.masked_fill(~topm_mask, -1e9)
                    else:
                        combined_scores = scores

                    if topk_q > 1:
                        k_eff = min(topk_q, combined_scores.shape[-1])
                        top_vals, top_idxs = torch.topk(combined_scores, k=k_eff, dim=-1)
                        weights = F.softmax(top_vals * 5.0, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
                        batch_idx = torch.arange(batch_size, device=pred_masks.device).unsqueeze(1).expand(-1, k_eff)
                        selected_masks = pred_masks[batch_idx, top_idxs].float()  # (B, K, H_m, W_m)
                        ens_prob = (torch.sigmoid(selected_masks) * weights).sum(dim=1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
                        best_masks = torch.log(ens_prob / (1.0 - ens_prob))
                    else:
                        best_idx = combined_scores.argmax(dim=-1)
                        batch_idx = torch.arange(batch_size, device=pred_masks.device)
                        best_masks = pred_masks[batch_idx, best_idx].unsqueeze(1)
            else:
                best_masks = pred_masks[:, 0:1]

            best_masks = F.interpolate(
                best_masks.float(),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
            result['pred_masks'] = best_masks
        else:
            result['pred_masks'] = torch.zeros(
                batch_size, 1, self.image_size, self.image_size,
                device=next(self.parameters()).device,
            )

        return result

    def _compute_fallback_loss(self, pred_masks, gt_masks):
        """Fallback Dice + BCE loss."""
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(), size=pred_masks.shape[-2:], mode='nearest'
            )

        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())

        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss

    @torch.no_grad()
    def predict(self, images, captions):
        """Inference-only forward pass."""
        self.eval()
        result = self.forward(images, captions)
        result['pred_probs'] = torch.sigmoid(result['pred_masks'])
        result['pred_binary'] = (result['pred_probs'] > 0.5).float()
        return result


