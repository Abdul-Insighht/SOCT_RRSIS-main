r"""
SOCT (Stochastic-Order Constrained Transport) Feature Alignment Module.
SOCT-SAM: Stochastic-Order Constrained Transport for Referring Remote Sensing Image Segmentation

Key Components:
1. Multi-scale Cost Lifting with Geometric Aggregation Operators (D_s).
2. Selective Target Demand, Area Prediction (\hat{\rho}), and Null Assignment (b_\emptyset).
3. Independent Contextual Anchor Localization (r_{mj}).
4. Adaptive Spatial Filtration with Bounded Bandwidth (\tau_m).
5. Dual-Conditioned Sinkhorn Solver with Directional Stochastic Ordering Constraints.
6. Target Occupancy Computation (q^{(s)}) & Target-Only Semantic Feature Fusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class GeometricAggregationOperator:
    """
    Continuous geometric aggregation operator D_s mapping spatial grids
    with column-sum area preservation: 1^T D_s = 1^T, D_f = I.
    Memory-efficient implementation: uses bilinear interpolation for adjoint lifting (D_s^T)
    and area adaptive pooling for forward aggregation (D_s), eliminating O(N_f * N_s) dense allocations.
    """
    @staticmethod
    def lift_cost(cos_dist: torch.Tensor, H_fine: int, W_fine: int, H_coarse: int, W_coarse: int) -> torch.Tensor:
        """
        Lifts coarse cost C^(s) to finest grid: D_s^T C^(s).
        cos_dist: (B, N_coarse, seq_len)
        Returns: (B, N_fine, seq_len)
        """
        if H_fine == H_coarse and W_fine == W_coarse:
            return cos_dist
        B, N_c, seq_len = cos_dist.shape
        cost_2d = cos_dist.permute(0, 2, 1).view(B, seq_len, H_coarse, W_coarse)
        lifted_2d = F.interpolate(cost_2d, size=(H_fine, W_fine), mode='bilinear', align_corners=False)
        return lifted_2d.flatten(2).permute(0, 2, 1)

    @staticmethod
    def aggregate_plan(P_fine: torch.Tensor, H_fine: int, W_fine: int, H_coarse: int, W_coarse: int) -> torch.Tensor:
        """
        Aggregates finest plan P* down to coarse scale s: D_s P*.
        P_fine: (B, N_fine, K)
        Returns: (B, N_coarse, K)
        """
        if H_fine == H_coarse and W_fine == W_coarse:
            return P_fine
        B, N_f, K = P_fine.shape
        P_2d = P_fine.permute(0, 2, 1).view(B, K, H_fine, W_fine)
        P_coarse_2d = F.adaptive_avg_pool2d(P_2d, (H_coarse, W_coarse))
        return P_coarse_2d.flatten(2).permute(0, 2, 1)


class DualConditionedSinkhorn(nn.Module):
    """
    Solves Stochastic-Order Constrained Transport via Dual Updates
    interleaved with entropic Sinkhorn transport.

    Primal: min <P, C> + eps * Entropy(P) + lambda_ord / 2 * sum_q w_q [g_q(P)]_+^2
    Dual update on multipliers delta_q >= 0 with adjusted cost:
    C_delta = C + sum_q delta_q B_q
    """
    def __init__(
        self,
        eps: float = 0.05,
        num_outer: int = 10,
        num_inner: int = 30,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.eps = eps
        self.num_outer = num_outer
        self.num_inner = num_inner
        self.dual_lr = dual_lr

    def _sinkhorn_inner(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        FP32 log-domain stabilized Sinkhorn algorithm.
        cost: (B, N, K)
        a: (B, N)
        b: (B, K)
        Returns: P (B, N, K) satisfying P 1 = a, P^T 1 = b
        """
        B, N, K = cost.shape
        log_a = torch.log(a.clamp(min=1e-12))
        log_b = torch.log(b.clamp(min=1e-12))

        # Scaling potentials in log-domain: u, v
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)

        eps = self.eps
        M = -cost / eps

        for _ in range(self.num_inner):
            # u = log_a - eps * logsumexp((v - cost)/eps)
            # In M terms: logsumexp(M + v[:, None, :], dim=-1)
            u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=-1)
            # v = log_b - eps * logsumexp((u - cost)/eps)
            v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

        log_P = M + u.unsqueeze(2) + v.unsqueeze(1)
        P = torch.exp(log_P).clamp(min=0.0)

        # Normalize to strictly satisfy marginals
        row_sum = P.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        P = P * (a.unsqueeze(-1) / row_sum)
        return P

    def forward(
        self,
        C_lifted: torch.Tensor,          # (B, N_f, K_cols)
        a: torch.Tensor,                 # (B, N_f)
        b: torch.Tensor,                 # (B, K_cols)
        target_col_mask: torch.Tensor,   # (B, K_cols) boolean mask for target tokens
        rho_hat: torch.Tensor,           # (B, 1) predicted target area
        omega: Optional[torch.Tensor],   # (B, M_rel, G+1, N_f) soft filtration weights
        F_A: Optional[torch.Tensor],     # (B, M_rel, G+1) anchor cumulative beliefs
        lambda_ord: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes dual-conditioned Sinkhorn.
        Returns:
            P_star: (B, N_f, K_cols) optimal transport plan
            order_penalty: scalar penalty value for logging/monitoring
        """
        B, N_f, K_cols = C_lifted.shape
        device = C_lifted.device

        # If no relations present or omega is None, fallback to ordinary selective OT
        if omega is None or F_A is None or omega.numel() == 0 or F_A.numel() == 0:
            P = self._sinkhorn_inner(C_lifted, a, b)
            return P, torch.tensor(0.0, device=device)

        # Flatten (m, ell) dimensions to q
        # omega: (B, M_rel, G_plus_1, N_f) -> (B, Q, N_f)
        B_dim, M_rel, G_p1, _ = omega.shape
        Q = M_rel * G_p1
        omega_flat = omega.view(B_dim, Q, N_f)         # (B, Q, N_f)
        F_A_flat = F_A.view(B_dim, Q)                  # (B, Q)

        # Weight per constraint
        w_q = 1.0 / max(1, Q)
        c_q = lambda_ord * w_q

        rho_safe = rho_hat.clamp(min=1e-4) # (B, 1)
        t_indicator = target_col_mask.float().unsqueeze(1) # (B, 1, K_cols)

        # Step 1: Solve for optimal dual variables delta* via projected gradient ascent
        # Run dual search without tracking graph across all 10 outer iterations for memory efficiency
        with torch.no_grad():
            delta = torch.zeros(B_dim, Q, device=device, dtype=torch.float32)
            for outer_it in range(self.num_outer):
                # Cost adjustment: C_delta = C + sum_q delta_q B_q
                delta_omega = torch.bmm(delta.unsqueeze(1), omega_flat).squeeze(1) # (B, N_f)
                delta_C = - (delta_omega / rho_safe).unsqueeze(-1) * t_indicator # (B, N_f, K_cols)

                C_adj = C_lifted + delta_C
                P = self._sinkhorn_inner(C_adj, a, b)

                # Compute target location distribution: p_j = (P t)_j / rho_hat
                P_target = (P * target_col_mask.unsqueeze(1)).sum(dim=-1) # (B, N_f)
                p_target = P_target / rho_safe # (B, N_f)

                # Target cumulative distribution along direction
                F_T = torch.bmm(omega_flat, p_target.unsqueeze(-1)).squeeze(-1) # (B, Q)
                g_q = F_A_flat - F_T # (B, Q)

                # Projected dual update
                step = self.dual_lr * (g_q - delta / c_q)
                delta = (delta + step).clamp(min=0.0)

        # Step 2: Differentiable unrolled solve with optimal delta*
        # Fixed-budget unrolling during training to propagate gradients to costs and demands
        delta_omega = torch.bmm(delta.unsqueeze(1), omega_flat).squeeze(1)
        delta_C = - (delta_omega / rho_safe).unsqueeze(-1) * t_indicator
        P_star = self._sinkhorn_inner(C_lifted + delta_C, a, b)

        # Residual order penalty for diagnostics
        P_target = (P_star * target_col_mask.unsqueeze(1)).sum(dim=-1)
        p_target = P_target / rho_safe
        F_T = torch.bmm(omega_flat, p_target.unsqueeze(-1)).squeeze(-1)
        g_q = F_A_flat - F_T
        order_penalty = 0.5 * lambda_ord * w_q * (F.relu(g_q) ** 2).sum(dim=-1).mean()

        return P_star, order_penalty


class SOCTAligner(nn.Module):
    """
    Main SOCT-SAM Alignment Engine:
    - Multi-scale cosine cost lifting
    - Target area prediction and null cost
    - Anchor belief localization
    - Adaptive spatial filtration
    - Dual-conditioned Sinkhorn
    - Target occupancy calculation & target-only feature fusion
    """
    def __init__(
        self,
        d_model: int = 256,
        num_scales: int = 3,
        lambda_ord: float = 5.0,
        theta_anchor: float = 0.1,
        ot_eps: float = 0.05,
        num_outer: int = 10,
        num_inner: int = 30,
        dual_lr: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.lambda_ord = lambda_ord
        self.theta_anchor = theta_anchor

        # Scale-specific visual projection heads
        self.visual_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])
        # Text projection head
        self.text_proj = nn.Linear(d_model, d_model)

        # Multi-scale cost combination weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Target Area Predictor Head H_rho([Pool(X^(f)); z]) -> (0, 1)
        self.area_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Within-target importance head h_k = H_t([T_k; z])
        self.within_target_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Null cost head: C_{j, null} = 2 * sigmoid(g_null([X_j; z]))
        self.null_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Adaptive Bandwidth Head H_tau([z; z_T; z_A; Pool(X^(f))]) -> (0, 1)
        self.bandwidth_head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Gated fusion MLP for each scale
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            for _ in range(num_scales)
        ])
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_scales)
        ])

        # Dual-Conditioned Sinkhorn Solver
        self.solver = DualConditionedSinkhorn(
            eps=ot_eps,
            num_outer=num_outer,
            num_inner=num_inner,
            dual_lr=dual_lr,
        )

        # Diagnostic state
        self.last_occupancy = None
        self.last_rho_hat = None
        self.last_tau = None
        self.last_order_penalty = None

    def forward(
        self,
        fpn_features: List[torch.Tensor],
        text_feats: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        parsed_batch: Optional[List[Dict]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of SOCT.

        Args:
            fpn_features: List of (B, C, H_s, W_s) for scales s=0 (finest) to S-1.
            text_feats: (B, seq_len, C) or (seq_len, B, C) contextual text tokens.
            text_mask: (B, seq_len) boolean mask (True = padding).
            parsed_batch: Optional list of parsed language roles per image in batch.

        Returns:
            aligned_fpn: List of enhanced FPN tensors.
            q_finest: (B, H_f, W_f) target occupancy map on finest grid.
            rho_hat: (B, 1) predicted target area.
            tau_m: Optional predicted bandwidths.
        """
        # Ensure batch-first for text: (B, seq_len, C)
        if text_feats.dim() == 3 and text_feats.shape[0] != len(fpn_features[0]):
            text_feats = text_feats.permute(1, 0, 2)
        B, seq_len, C = text_feats.shape
        device = text_feats.device

        # Pooled sentence embedding z: (B, C)
        if text_mask is not None:
            valid_mask = (~text_mask).float().unsqueeze(-1)
            z = (text_feats * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            z = text_feats.mean(dim=1)

        # Extract finest scale feature grid f = 0
        feat_fine = fpn_features[0]
        if hasattr(feat_fine, 'tensors'):
            feat_fine = feat_fine.tensors
        B, C_feat, H_f, W_f = feat_fine.shape
        N_f = H_f * W_f

        # Pool(X^(f)): (B, C)
        pool_fine = feat_fine.mean(dim=[-2, -1])

        # 1. Predict Target Area \hat{\rho} \in (0, 1)
        area_input = torch.cat([pool_fine, z], dim=-1)
        rho_hat = self.area_head(area_input).clamp(min=1e-4, max=1.0 - 1e-4) # (B, 1)
        self.last_rho_hat = rho_hat

        # 2. Compute Projected Cosine Distance Matrix for each scale
        txt_proj = F.normalize(self.text_proj(text_feats), dim=-1) # (B, seq_len, C)

        scale_costs = []
        for s, feat in enumerate(fpn_features):
            t_feat = feat.tensors if hasattr(feat, 'tensors') else feat
            Bs, Cs, Hs, Ws = t_feat.shape
            vis_flat = t_feat.flatten(2).permute(0, 2, 1) # (B, N_s, C)
            proj_vis = F.normalize(self.visual_projs[min(s, len(self.visual_projs)-1)](vis_flat), dim=-1)

            # Cosine distance: 1 - <X_s, T>
            cos_dist = 1.0 - torch.bmm(proj_vis, txt_proj.transpose(1, 2)) # (B, N_s, seq_len)
            scale_costs.append((cos_dist, Hs, Ws))

        # 3. Geometric Cost Lifting to Finest Grid
        w_scales = F.softmax(self.scale_weights, dim=0)
        C_lifted_tokens = torch.zeros(B, N_f, seq_len, device=device, dtype=torch.float32)

        for s, (cos_dist, Hs, Ws) in enumerate(scale_costs):
            lifted_s = GeometricAggregationOperator.lift_cost(cos_dist, H_f, W_f, Hs, Ws)
            C_lifted_tokens = C_lifted_tokens + w_scales[s] * lifted_s

        # 4. Null Column Cost: C_{j, null} = 2 * sigmoid(g_null([X_j^(f); z]))
        fine_flat = feat_fine.flatten(2).permute(0, 2, 1) # (B, N_f, C)
        z_expanded = z.unsqueeze(1).expand(-1, N_f, -1)   # (B, N_f, C)
        null_input = torch.cat([fine_flat, z_expanded], dim=-1)
        C_null = 2.0 * torch.sigmoid(self.null_head(null_input)) # (B, N_f, 1)

        # Full cost matrix including Null column as last column: (B, N_f, seq_len + 1)
        C_full = torch.cat([C_lifted_tokens, C_null], dim=-1)

        # 5. Process Language Roles (Target vs Anchors vs Null)
        # Construct marginal demands a (B, N_f) and b (B, seq_len + 1)
        a = torch.full((B, N_f), 1.0 / N_f, device=device, dtype=torch.float32)

        # Within-target logits
        z_tok_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        target_h = self.within_target_head(torch.cat([text_feats, z_tok_expanded], dim=-1)).squeeze(-1) # (B, seq_len)

        target_col_mask = torch.zeros(B, seq_len + 1, device=device, dtype=torch.bool)
        b_marginal = torch.zeros(B, seq_len + 1, device=device, dtype=torch.float32)

        # Precompute fine grid coordinates for filtration
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / H_f, 1.0 - 0.5 / H_f, H_f, device=device),
            torch.linspace(0.5 / W_f, 1.0 - 0.5 / W_f, W_f, device=device),
            indexing='ij'
        )
        # r_pos: (N_f, 2), columns: (x, y)
        r_pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        G = min(32, max(H_f, W_f))
        u_eval = torch.linspace(0.0, 1.0, G + 1, device=device) # (G+1,)
        tau_0 = 1.0 / G
        tau_min = tau_0 / 2.0
        tau_max = tau_0 * 2.0

        all_omega = []
        all_F_A = []
        tau_recorded = []

        direction_vecs = {
            'left': torch.tensor([1.0, 0.0], device=device),
            'right': torch.tensor([-1.0, 0.0], device=device),
            'above': torch.tensor([0.0, 1.0], device=device),
            'below': torch.tensor([0.0, -1.0], device=device),
        }

        for b_idx in range(B):
            # Target indices for this sample
            if parsed_batch is not None and b_idx < len(parsed_batch):
                t_indices = parsed_batch[b_idx].get('target_indices', list(range(seq_len)))
                relations = parsed_batch[b_idx].get('relations', [])
            else:
                t_indices = list(range(seq_len))
                relations = []

            # Boundary-safe filtering for target indices
            t_indices = [idx for idx in t_indices if 0 <= idx < seq_len]
            if text_mask is not None:
                t_indices = [idx for idx in t_indices if not text_mask[b_idx, idx].item()]
            if not t_indices:
                valid_toks = [i for i in range(seq_len) if text_mask is None or not text_mask[b_idx, i].item()]
                t_indices = [valid_toks[0]] if valid_toks else [0]

            for idx in t_indices:
                target_col_mask[b_idx, idx] = True

            # Target weights alpha_k
            t_logits = target_h[b_idx, t_indices]
            alpha = F.softmax(t_logits, dim=0)

            # Assign demands: b_k = rho_hat * alpha_k
            for k_local, idx in enumerate(t_indices):
                b_marginal[b_idx, idx] = rho_hat[b_idx, 0] * alpha[k_local]

            # Null column demand: b_null = 1 - rho_hat
            b_marginal[b_idx, -1] = 1.0 - rho_hat[b_idx, 0]

            # Process relations for this sample
            sample_omega = []
            sample_F_A = []
            for a_indices, direction in relations:
                # Boundary-safe filtering for anchor indices
                a_indices = [idx for idx in a_indices if 0 <= idx < seq_len]
                if text_mask is not None:
                    a_indices = [idx for idx in a_indices if not text_mask[b_idx, idx].item()]
                if not a_indices:
                    continue
                d_key = direction.lower()
                n_dir = direction_vecs.get(d_key, direction_vecs['left'])

                # Anchor spatial prior: r_m from averaged lifted costs
                # C_lifted_tokens: (B, N_f, seq_len)
                c_A = C_lifted_tokens[b_idx, :, a_indices].mean(dim=-1) # (N_f,)
                # Gibbs location belief: r_{mj} = a_j * exp(-c_{mj}^A / theta_A) / sum(...)
                log_belief = -c_A / max(1e-4, self.theta_anchor)
                r_m = F.softmax(log_belief, dim=0) # (N_f,)

                # Target & Anchor pooled representations
                z_T = text_feats[b_idx, t_indices].mean(dim=0)
                z_A = text_feats[b_idx, a_indices].mean(dim=0)

                # Predict adaptive bandwidth tau_m
                bw_in = torch.cat([z[b_idx], z_T, z_A, pool_fine[b_idx]], dim=-1).unsqueeze(0)
                tau_raw = self.bandwidth_head(bw_in).squeeze()
                tau_m = tau_min + (tau_max - tau_min) * tau_raw
                tau_recorded.append(tau_m)

                # Normalized projection h_mj: n_m^T r_j
                proj_h = torch.mv(r_pos, n_dir) # (N_f,)
                h_min = proj_h.min()
                h_max = proj_h.max()
                h_norm = (proj_h - h_min) / (h_max - h_min).clamp(min=1e-6) # (N_f,)

                # Soft spatial filtration: omega_{m, ell, j} = sigmoid((u_ell - h_j) / tau_m)
                # (G+1, 1) - (1, N_f) -> (G+1, N_f)
                diff = (u_eval.unsqueeze(1) - h_norm.unsqueeze(0)) / tau_m.clamp(min=1e-4)
                omega_m = torch.sigmoid(diff) # (G+1, N_f)

                # Anchor cumulative belief: F_A_{m, ell} = sum_j r_{mj} * omega_{m, ell, j}
                F_A_m = torch.mv(omega_m, r_m) # (G+1,)

                sample_omega.append(omega_m)
                sample_F_A.append(F_A_m)

            if sample_omega:
                all_omega.append(torch.stack(sample_omega, dim=0)) # (M, G+1, N_f)
                all_F_A.append(torch.stack(sample_F_A, dim=0))     # (M, G+1)
            else:
                all_omega.append(None)
                all_F_A.append(None)

        # Batch collation for solver
        # If any sample has relations, pad to max M
        max_M = max([o.shape[0] if o is not None else 0 for o in all_omega])
        batched_omega = None
        batched_F_A = None

        if max_M > 0:
            batched_omega = torch.zeros(B, max_M, G + 1, N_f, device=device, dtype=torch.float32)
            batched_F_A = torch.zeros(B, max_M, G + 1, device=device, dtype=torch.float32)
            for b_idx in range(B):
                if all_omega[b_idx] is not None:
                    m_cur = all_omega[b_idx].shape[0]
                    batched_omega[b_idx, :m_cur] = all_omega[b_idx]
                    batched_F_A[b_idx, :m_cur] = all_F_A[b_idx]

        # 6. Execute Dual-Conditioned Sinkhorn Solver
        P_star, order_penalty = self.solver(
            C_lifted=C_full,
            a=a,
            b=b_marginal,
            target_col_mask=target_col_mask,
            rho_hat=rho_hat,
            omega=batched_omega,
            F_A=batched_F_A,
            lambda_ord=self.lambda_ord,
        ) # (B, N_f, seq_len + 1)
        self.last_order_penalty = order_penalty

        # 7. Compute Target Occupancy on Finest Grid: q^(f) = diag(a)^(-1) * P* t
        # a_j = 1/N_f -> diag(a)^(-1) = N_f
        # P_star[:, :, :-1] contains text tokens; last column is null
        P_targets_only = P_star * target_col_mask.unsqueeze(1)
        q_finest_flat = P_targets_only.sum(dim=-1) * N_f # (B, N_f)
        q_finest = q_finest_flat.view(B, H_f, W_f).clamp(min=0.0, max=1.0)
        self.last_occupancy = q_finest

        # 8. Feature Fusion across scales using Target-Only transport values
        # P^(s) = D_s P* -> Target value A^(s) = diag(a^(s))^(-1) D_s (P*_T T_T^T)
        aligned_fpn = []
        P_target_fine = P_targets_only[:, :, :-1] # (B, N_f, seq_len), exclude null
        for s, feat in enumerate(fpn_features):
            t_feat = feat.tensors if hasattr(feat, 'tensors') else feat
            Bs, Cs, Hs, Ws = t_feat.shape
            Ns = Hs * Ws

            # Aggregate fine plan target slice to coarse: P_target_s = D_s @ P_target_fine
            P_target_s = GeometricAggregationOperator.aggregate_plan(P_target_fine, H_f, W_f, Hs, Ws) # (B, N_s, seq_len)

            # Target transported values: A^(s) = N_s * P_target_s @ T
            A_s = torch.bmm(P_target_s * Ns, text_feats) # (B, N_s, C)
            A_proj = self.output_projs[min(s, len(self.output_projs)-1)](A_s)

            # Residual Gated Fusion
            feat_flat = t_feat.flatten(2).permute(0, 2, 1) # (B, N_s, C)
            gate_in = torch.cat([feat_flat, A_proj], dim=-1)
            gate_w = self.fusion_gates[min(s, len(self.fusion_gates)-1)](gate_in)

            enhanced_flat = feat_flat + gate_w * A_proj
            enhanced_norm = self.layer_norms[min(s, len(self.layer_norms)-1)](enhanced_flat)
            enhanced = enhanced_norm.permute(0, 2, 1).view(Bs, Cs, Hs, Ws)

            if hasattr(feat, 'tensors'):
                feat.tensors = enhanced
                aligned_fpn.append(feat)
            else:
                aligned_fpn.append(enhanced)

        tau_ret = torch.stack(tau_recorded).mean() if tau_recorded else None
        self.last_tau = tau_ret

        return aligned_fpn, q_finest, rho_hat, tau_ret
