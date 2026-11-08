r"""
Grounding-Aware Prompt Generation from Target Occupancy for SOCT-SAM.
SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation

- Derive geometric prompts directly from finest-grid target occupancy q^(f) \in [0, 1].
- Select spatially separated local maxima (peaks) of q^(f) as positive points (cap K = 5).
- Do not use max_k P_{jk} or min-max normalization.
- Retain only one point when predicted target area \hat{\rho} < 0.01 (tiny-target regime).
- Map cell centres accurately to image prompt coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GroundingAwarePromptGenerator(nn.Module):
    """
    Generates sparse geometric point prompts for SAM3 from target occupancy maps.
    """
    def __init__(self, max_points: int = 5, min_separation: int = 2):
        super().__init__()
        self.max_points = max_points
        self.min_separation = min_separation
        print(f"[SOCT-PromptGenerator] Initialized with max_points={max_points}, min_sep={min_separation}")

    @torch.no_grad()
    def forward(
        self,
        occupancy: torch.Tensor,
        original_image_size: int,
        device: torch.device,
        rho_hat: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            occupancy: (B, H_f, W_f) target occupancy map q^(f) in [0, 1],
                       OR (B, HW, seq) raw transport plan (for backward compatibility).
            original_image_size: int, spatial size of image (e.g. 504).
            device: torch.device.
            rho_hat: Optional (B, 1) predicted target area from area head.
            text_mask: Optional (B, seq) boolean mask for backward compat.

        Returns:
            points: (B, K_max, 2) coordinates in image space [x, y].
            points_mask: (B, K_max) boolean mask (False = valid, True = ignore/padded).
            point_labels: (B, K_max) long tensor (1 = foreground positive point, 0 = ignore).
        """
        # Handle input shape
        if occupancy.dim() == 3 and occupancy.shape[1] != occupancy.shape[2]:
            # Raw P of shape (B, HW, seq) -> backward compatibility fallback
            B, HW, seq = occupancy.shape
            H = W = int(HW ** 0.5)
            if text_mask is not None:
                valid = (~text_mask).unsqueeze(1).float()
                q_map = (occupancy * valid).sum(dim=-1).view(B, H, W)
            else:
                q_map = occupancy.sum(dim=-1).view(B, H, W)
        elif occupancy.dim() == 3:
            # Proper occupancy map (B, H_f, W_f)
            B, H, W = occupancy.shape
            q_map = occupancy
        else:
            raise ValueError(f"Unexpected occupancy shape: {occupancy.shape}")

        K_max = self.max_points
        points = torch.zeros((B, K_max, 2), device=device, dtype=torch.float32)
        points_mask = torch.zeros((B, K_max), device=device, dtype=torch.bool)
        point_labels = torch.ones((B, K_max), device=device, dtype=torch.long)

        # Scale factors to convert feature cell indices to image pixels
        scale_y = float(original_image_size) / float(H)
        scale_x = float(original_image_size) / float(W)

        # Spatially separated peak extraction via 3x3 local maximum suppression
        # local_max: (B, 1, H, W)
        kernel_size = 2 * self.min_separation + 1
        padding = self.min_separation
        q_4d = q_map.unsqueeze(1) # (B, 1, H, W)
        local_max = F.max_pool2d(q_4d, kernel_size=kernel_size, stride=1, padding=padding)
        is_peak = (q_4d == local_max) & (q_4d > 0.05) # Peak threshold

        for b in range(B):
            sample_q = q_map[b] # (H, W)
            sample_peaks = is_peak[b, 0] # (H, W) bool

            # Coordinates of peaks sorted by occupancy descending
            peak_coords = torch.nonzero(sample_peaks, as_tuple=False) # (N_peaks, 2) with (y, x)
            if peak_coords.shape[0] > 0:
                peak_vals = sample_q[peak_coords[:, 0], peak_coords[:, 1]]
                sorted_indices = torch.argsort(peak_vals, descending=True)
                sorted_coords = peak_coords[sorted_indices]
            else:
                # Fallback: argmax of map if no peaks exceed threshold
                flat_idx = torch.argmax(sample_q.flatten())
                y_single = torch.div(flat_idx, W, rounding_mode='floor')
                x_single = flat_idx % W
                sorted_coords = torch.tensor([[y_single, x_single]], device=device)

            # Determine how many points to keep based on predicted area \hat{\rho}
            if rho_hat is not None:
                area_ratio = rho_hat[b, 0].item()
            else:
                area_ratio = (sample_q > 0.5).sum().float().item() / (H * W)

            # Scale-aware prompting (Paper Sec 3.7):
            # Retain 1 focal centroid point for truly tiny targets (area_ratio < 0.01) to prevent noise.
            # For compact targets, use up to 3 points. For larger structures, use up to K_max=5 peaks.
            if area_ratio < 0.01:
                n_keep = 1
            elif area_ratio < 0.05:
                n_keep = min(3, sorted_coords.shape[0])
            else:
                n_keep = min(K_max, sorted_coords.shape[0])

            for k in range(n_keep):
                y_c = sorted_coords[k, 0].float()
                x_c = sorted_coords[k, 1].float()
                # Center of cell mapped to image coordinates (x, y)
                points[b, k, 0] = (x_c + 0.5) * scale_x
                points[b, k, 1] = (y_c + 0.5) * scale_y

            # Pad remaining points
            if n_keep < K_max:
                points_mask[b, n_keep:] = True
                point_labels[b, n_keep:] = 0

        return points, points_mask, point_labels
