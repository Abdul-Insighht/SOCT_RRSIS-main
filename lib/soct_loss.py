r"""
SOCT Loss Functions for SOCT-SAM.
SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation

1. Occupancy Loss: Balanced BCE + Soft Dice on finest grid target occupancy q^(f) vs ground truth.
2. Target Area Loss: Huber loss on log-scale predicted area \hat{\rho} vs GT area \rho*.
3. Bandwidth Regularization: Quadratic penalty on log(\tau_m / \tau_0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SOCTSupervisionLoss(nn.Module):
    """
    Computes SOCT-specific auxiliary supervision losses:
    - L_occ: Occupancy supervision (Balanced BCE + Dice)
    - L_rho: Target Area Huber supervision
    - L_tau: Bandwidth regularization
    """
    def __init__(
        self,
        lambda_occ: float = 0.3,
        lambda_rho: float = 0.2,
        lambda_tau: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_occ = lambda_occ
        self.lambda_rho = lambda_rho
        self.lambda_tau = lambda_tau
        self.eps = eps

    def forward(
        self,
        q_finest: torch.Tensor,        # (B, H_f, W_f) target occupancy map
        rho_hat: torch.Tensor,         # (B, 1) predicted target area ratio
        gt_masks: torch.Tensor,        # (B, 1, H, W) binary ground truth mask
        tau_val: Optional[torch.Tensor] = None, # scalar or tensor of bandwidths
        tau_0: Optional[float] = None, # reference bandwidth (1 / G)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_soct_loss: weighted combination of L_occ, L_rho, L_tau
            L_occ: Occupancy loss
            L_rho: Area loss
            L_tau: Bandwidth loss
        """
        device = q_finest.device
        B, H_f, W_f = q_finest.shape
        N_f = H_f * W_f

        # Downsample GT mask to finest grid size (area-averaged labels y_j \in [0, 1])
        # gt_masks: (B, 1, H_img, W_img)
        y_gt = F.interpolate(gt_masks.float(), size=(H_f, W_f), mode='area').squeeze(1) # (B, H_f, W_f)
        y_flat = y_gt.flatten(1) # (B, N_f)
        q_flat = q_finest.flatten(1) # (B, N_f)

        # Uniform valid cell area a_j = 1 / N_f
        # 1. Balanced BCE Loss (Eq. 26)
        delta = self.eps
        # Positive term:
        pos_weight = y_flat.sum(dim=-1, keepdim=True) + delta # (B, 1)
        neg_weight = (1.0 - y_flat).sum(dim=-1, keepdim=True) + delta # (B, 1)

        pos_bce = (y_flat * torch.log(q_flat.clamp(min=delta))).sum(dim=-1, keepdim=True) / pos_weight
        neg_bce = ((1.0 - y_flat) * torch.log((1.0 - q_flat).clamp(min=delta))).sum(dim=-1, keepdim=True) / neg_weight
        bal_bce = -0.5 * (pos_bce + neg_bce).mean()

        # Soft Dice Loss (Eq. 22 second term)
        intersection = (q_flat * y_flat).sum(dim=-1)
        union = q_flat.sum(dim=-1) + y_flat.sum(dim=-1)
        dice_loss = (1.0 - (2.0 * intersection + delta) / (union + delta)).mean()

        L_occ = bal_bce + dice_loss

        # 2. Target Area Loss (Eq. 23)
        # rho* = GT area ratio in [0, 1]
        rho_star = gt_masks.float().mean(dim=[-2, -1]).view(B, 1) # (B, 1)
        # delta_rho = 1 valid image pixel's area
        img_h, img_w = gt_masks.shape[-2:]
        delta_rho = 1.0 / float(img_h * img_w)

        log_rho_pred = torch.log(rho_hat + delta_rho)
        log_rho_gt = torch.log(rho_star + delta_rho)
        L_rho = F.smooth_l1_loss(log_rho_pred, log_rho_gt, beta=1.0)

        # 3. Bandwidth Regularization (Eq. 24)
        if tau_val is not None and tau_0 is not None and tau_0 > 0:
            tau_ratio = torch.log(tau_val / tau_0)
            L_tau = (tau_ratio ** 2).mean()
        else:
            L_tau = torch.tensor(0.0, device=device)

        total_soct_loss = (
            self.lambda_occ * L_occ +
            self.lambda_rho * L_rho +
            self.lambda_tau * L_tau
        )

        return total_soct_loss, L_occ, L_rho, L_tau
