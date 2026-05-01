# -*- coding: utf-8 -*-
from typing import Dict, Optional, NamedTuple, Tuple
import torch
from .loss import TopoLossCOD


class TopoOut(NamedTuple):
    weighted_loss: torch.Tensor
    log_str: str
    parts: Dict[str, torch.Tensor]
    alpha: float
    topo_coef: float


def _cosine_ramp(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return 0.5 * (1 - torch.cos(torch.tensor(x * 3.1415926535))).item()


def topo_addon(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topo: TopoLossCOD,
    sod_loss_for_balance: torch.Tensor,
    lambda_topo: float = 1e-1,
    iter_percentage: float = 1.0,
    ramp_milestones: Tuple[float, float] = (0.30, 0.80),
    grad_balance: bool = True,
    amp_device: str = "cuda",
    use_ortho: bool = False,
    ortho_mode: str = "sum",
    ortho_terms: Tuple[str, ...] = ("ph",),
    conflict_only: bool = True,
    eps_pix_norm: float = 1e-6,
) -> Optional[TopoOut]:
    topo_coef = _cosine_ramp(float(iter_percentage), *ramp_milestones)
    if topo_coef <= 0.0:
        return None

    def _to_probs(x: torch.Tensor) -> torch.Tensor:
        xm, xM = float(x.min()), float(x.max())
        return x if (-1e-6 <= xm and xM <= 1.0 + 1e-6) else torch.sigmoid(x)

    with torch.no_grad():
        teacher_probs = _to_probs(teacher_logits).detach()
        mu = student_logits.mean()
        std = student_logits.std().clamp_min(1e-6)
    z = (student_logits - mu) / std

    with torch.amp.autocast(amp_device, enabled=False):
        topo_L_raw, topo_parts = topo(z.float(), ph_target_probs=teacher_probs)

    if not isinstance(topo_parts, dict):
        topo_parts = {}

    if not use_ortho:
        alpha_val = 1.0
        weighted = lambda_topo * topo_coef * topo_L_raw
        if grad_balance:
            try:
                g_pix = torch.autograd.grad(sod_loss_for_balance, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
                g_top = torch.autograd.grad(topo_L_raw, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
                if (g_pix is not None) and (g_top is not None):
                    num = g_pix.norm()
                    den = g_top.norm().clamp_min(1e-8)
                    alpha = (num / den).detach().pow(0.5).clamp(3e-2, 1e+1)
                    alpha_val = float(alpha.item())
                    weighted = lambda_topo * topo_coef * (alpha * topo_L_raw)
            except Exception:
                pass
        log_str = f"topo(λ={lambda_topo:.1e}*{topo_coef:.2f}*α={alpha_val:.2e}): {float(weighted.item()):.5f}"
        return TopoOut(weighted_loss=weighted, log_str=log_str, parts=topo_parts, alpha=alpha_val, topo_coef=float(topo_coef))

    def _project_surrogate_conditional(scalar: torch.Tensor, g_pix_cached: Optional[torch.Tensor]) -> torch.Tensor:
        cap_ratio = 0.25
        if g_pix_cached is None:
            return scalar
        batch_size = g_pix_cached.shape[0]
        gp_flat = g_pix_cached.detach().reshape(batch_size, -1)
        gp_n = gp_flat.norm(dim=1).clamp_min(eps_pix_norm)
        if bool((gp_n <= eps_pix_norm).all().item()):
            return scalar
        g_top = torch.autograd.grad(scalar, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
        if g_top is None:
            return scalar
        gt_flat = g_top.detach().reshape(batch_size, -1)
        gt_n = gt_flat.norm(dim=1).clamp_min(1e-12)
        dot = (gp_flat * gt_flat).sum(dim=1)
        conflict = (dot < 0) if conflict_only else torch.ones_like(dot, dtype=torch.bool)
        if conflict_only and (conflict.float().mean().item() == 0.0):
            return scalar
        proj_coeff = (dot / (gp_n * gp_n).clamp_min(1e-12)).view(batch_size, 1)
        proj = proj_coeff * gp_flat
        g_ortho_flat = gt_flat - proj
        go_n = g_ortho_flat.norm(dim=1).clamp_min(1e-12)
        cap = (cap_ratio * gp_n).clamp_min(1e-12)
        scale = (cap / go_n).clamp(max=1.0)
        g_ortho_flat = g_ortho_flat * scale.view(batch_size, 1)
        g_ortho = g_ortho_flat.view_as(g_top)
        g_use = torch.where(conflict.view(batch_size, 1, 1, 1), g_ortho, g_top)
        return (g_use.detach() * student_logits).sum(dim=(1, 2, 3)).mean()

    try:
        g_pix = torch.autograd.grad(sod_loss_for_balance, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
    except Exception:
        g_pix = None

    if ortho_mode == "per_term":
        per = topo_parts.get("_per_term", None)
        lam_map = {"ph": float(getattr(topo, "lambda_ph", 1.0))}
        if isinstance(per, dict) and len(per) > 0:
            l_acc = student_logits.new_tensor(0.0)
            for name, scalar in per.items():
                lam = lam_map.get(name, 0.0)
                if lam == 0.0:
                    continue
                if name in ortho_terms:
                    l_acc = l_acc + lam * _project_surrogate_conditional(scalar, g_pix)
                else:
                    l_acc = l_acc + lam * scalar
            l_sur = l_acc
        else:
            l_sur = _project_surrogate_conditional(topo_L_raw, g_pix)
    else:
        l_sur = _project_surrogate_conditional(topo_L_raw, g_pix)

    alpha_val = 1.0
    weighted = lambda_topo * topo_coef * l_sur
    if grad_balance:
        try:
            if g_pix is None:
                g_pix = torch.autograd.grad(sod_loss_for_balance, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
            g_top = torch.autograd.grad(l_sur, student_logits, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if (g_pix is not None) and (g_top is not None):
                num = g_pix.norm()
                den = g_top.norm().clamp_min(1e-8)
                alpha = (num / den).detach().pow(0.5).clamp(3e-2, 1e+1)
                alpha_val = float(alpha.item())
                weighted = lambda_topo * topo_coef * (alpha * l_sur)
        except Exception:
            pass
    log_str = f"topo[ortho-{ortho_mode}](λ={lambda_topo:.1e}*{topo_coef:.2f}*α={alpha_val:.2e}): {float(weighted.item()):.5f}"
    return TopoOut(weighted_loss=weighted, log_str=log_str, parts=topo_parts, alpha=alpha_val, topo_coef=float(topo_coef))
