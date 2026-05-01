# -*- coding: utf-8 -*-
from typing import Tuple
import torch


def to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:  # (H,W) -> (1,1,H,W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:  # (B,H,W) -> (B,1,H,W)
        x = x.unsqueeze(1)
    return x

def soft_counts_4n(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = to_bchw(p).clamp(0, 1)               # (B,1,H,W)
    F_cnt = p.sum(dim=(2,3))                 # [B,1]
    ex = (p[..., :, :-1] * p[..., :, 1:]).sum(dim=(2,3))
    ey = (p[..., :-1, :] * p[..., 1:, :]).sum(dim=(2,3))
    E_cnt = ex + ey                          # [B,1]
    V_cnt = (p[..., :-1, :-1] * p[..., :-1, 1:] * p[..., 1:, :-1] * p[..., 1:, 1:]).sum(dim=(2,3))
    return V_cnt, E_cnt, F_cnt
