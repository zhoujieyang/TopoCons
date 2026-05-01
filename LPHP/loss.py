# -*- coding: utf-8 -*-
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from .ph import PHProxyLoss


class TopoLossCOD(nn.Module):
    def __init__(
        self,
        lambda_ph: float = 1.0,
        *,
        ph_num_thresholds: int = 16,
        ph_mode: str = "cap",
        ph_downsample: int = 4,
        ph_max_events: int = 12,
        ph_compute_every_n: int = 1,
        ph_weight_temp: float = 0.7,
        ph_chunk_size: int = 8,
        ph_tgt_smooth_ks: int = 3,
        ph_importance_mix: str = "auto",
        ph_lambda_mass: float = 1.0,
        ph_lambda_pos: float = 0.5,
        ph_use_uncert_gate: bool = False,
        ph_gate_detach: bool = True,
        ph_norm: str = "sqrt",
        ph_huber_delta: float = 0.25,
        ph_simp_weight_when_uncertain: float = 0.25,
        ph_pos_weight_when_uncertain: float = 0.3,
        ph_conf_thr_auto_mix: float = 0.6,
        ph_hard_tgt_u_thr: float = 0.02,
    ):
        super().__init__()
        self.lambda_ph = float(lambda_ph)
        # kept only for interface compatibility with ortho_mode="per_term"
        self.ph = PHProxyLoss(
            num_thresholds=ph_num_thresholds,
            mode=ph_mode,
            downsample=ph_downsample,
            max_events=ph_max_events,
            compute_every_n=ph_compute_every_n,
            weight_temp=ph_weight_temp,
            chunk_size=ph_chunk_size,
            tgt_smooth_ks=ph_tgt_smooth_ks,
            importance_mix=ph_importance_mix,
            lambda_mass=ph_lambda_mass,
            lambda_pos=ph_lambda_pos,
            use_uncert_gate=ph_use_uncert_gate,
            gate_detach=ph_gate_detach,
            norm=ph_norm,
            huber_delta=ph_huber_delta,
            simp_weight_when_uncertain=ph_simp_weight_when_uncertain,
            pos_weight_when_uncertain=ph_pos_weight_when_uncertain,
            conf_thr_auto_mix=ph_conf_thr_auto_mix,
            hard_tgt_u_thr=ph_hard_tgt_u_thr,
        )

    def forward(
        self,
        logits: torch.Tensor,
        *,
        gt_mask: Optional[torch.Tensor] = None,
        ph_target_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts: Dict[str, torch.Tensor] = {}
        total = logits.new_tensor(0.0)
        probs = torch.sigmoid(logits)
        if self.lambda_ph > 0:
            ph_tgt = ph_target_probs if ph_target_probs is not None else (
                gt_mask.float() if gt_mask is not None else probs.detach()
            )
            l_ph_raw, pph = self.ph(probs, ph_tgt)
            parts.update(pph)
            parts["loss/ph_raw"] = l_ph_raw.detach()
            parts.setdefault("_per_term", {})["ph"] = l_ph_raw
            total = total + self.lambda_ph * l_ph_raw
        return total, parts
