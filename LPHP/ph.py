# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_bchw, soft_counts_4n


class PHProxyLoss(nn.Module):
    def __init__(
        self,
        num_thresholds: int = 16,
        *,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        sigma: float = 0.08,
        downsample: int = 4,
        num_dirs: int = 4,
        p: float = 1.0,
        lambda_mass: float = 1.0,
        lambda_pos: float = 0.5,
        weight_temp: float = 0.7,
        max_events: int = 12,
        use_uncert_gate: bool = True,
        gate_detach: bool = True,
        compute_every_n: int = 1,
        norm: str = "sqrt",
        huber_delta: float = 0.25,
        chunk_size: int = 8,
        tgt_smooth_ks: int = 3,
        importance_mix: str = "auto",
        mode: str = "cap",
        conf_thr_auto_mix: float = 0.6,
        simp_weight_when_uncertain: float = 0.25,
        pos_weight_when_uncertain: float = 0.3,
        hard_tgt_u_thr: float = 0.02,
    ):
        super().__init__()
        self.M = int(num_thresholds)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.sigma = float(sigma)
        self.downsample = int(downsample)
        self.K = int(num_dirs)
        self.p = float(p)
        self.lambda_mass = float(lambda_mass)
        self.lambda_pos = float(lambda_pos)
        self.weight_temp = float(weight_temp)
        self.max_events = int(max_events) if max_events is not None else -1
        self.use_uncert_gate = bool(use_uncert_gate)
        self.gate_detach = bool(gate_detach)
        self.compute_every_n = int(compute_every_n)
        self.norm = str(norm)
        self.huber_delta = float(huber_delta)
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0
        self.tgt_smooth_ks = int(tgt_smooth_ks)
        self.importance_mix = str(importance_mix)
        self.mode = str(mode)
        self.conf_thr_auto_mix = float(conf_thr_auto_mix)
        self.simp_weight_when_uncertain = float(simp_weight_when_uncertain)
        self.pos_weight_when_uncertain = float(pos_weight_when_uncertain)
        self.hard_tgt_u_thr = float(hard_tgt_u_thr)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._warmup_steps = 800
        self._temp_warmup_factor = 1.5
        self._cached_shape = None
        self._grid_device = None
        self._heights = None

    @torch.no_grad()
    def _ensure_heights(self, device, H: int, W: int, dtype: torch.dtype):
        if self._cached_shape == (H, W) and self._grid_device == str(device) and self._heights is not None and self._heights.dtype == dtype:
            return
        tH = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        tW = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        try:
            yy, xx = torch.meshgrid(tH, tW, indexing="ij")
        except TypeError:
            yy, xx = torch.meshgrid(tH, tW)
        grid_y = yy[None]
        grid_x = xx[None]
        if self.K <= 1:
            heights = grid_x
        else:
            angles = torch.linspace(0.0, math.pi, steps=self.K, device=device, dtype=dtype)
            hs = []
            for a in angles:
                hs.append(float(torch.cos(a).item()) * grid_x + float(torch.sin(a).item()) * grid_y)
            heights = torch.cat(hs, dim=0).contiguous()
        self._heights = heights
        self._cached_shape = (H, W)
        self._grid_device = str(device)

    def _chi_from_mask(self, m: torch.Tensor) -> torch.Tensor:
        V, E, F_ = soft_counts_4n(m)
        return (V - E + F_).squeeze(1)

    def _huber(self, x: torch.Tensor) -> torch.Tensor:
        ax = x.abs()
        d = self.huber_delta
        return torch.where(ax < d, 0.5 * ax * ax / d, ax - 0.5 * d)

    @torch.no_grad()
    def _maybe_smooth_target(self, p_tgt: torch.Tensor) -> torch.Tensor:
        ks = self.tgt_smooth_ks
        if ks is None or ks <= 1:
            return p_tgt
        u = (p_tgt * (1.0 - p_tgt)).mean().item()
        if u < self.hard_tgt_u_thr:
            return p_tgt
        pad = ks // 2
        return F.avg_pool2d(p_tgt, kernel_size=ks, stride=1, padding=pad)

    def forward(self, probs_pred: torch.Tensor, probs_tgt: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if self.training:
            self._step += 1
            if self.compute_every_n > 1 and int(self._step.item()) % self.compute_every_n != 0:
                z = probs_pred.new_tensor(0.0)
                return z, {"ph/skip": torch.tensor(1.0, device=z.device)}

        warm_coef = (self._step.float() / float(max(1, self._warmup_steps))).clamp(0.0, 1.0) if self.training else probs_pred.new_tensor(1.0)
        p_pred = to_bchw(probs_pred)
        p_tgt = to_bchw(probs_tgt)
        p_pred = torch.sigmoid(p_pred) if (p_pred.min() < -0.1 or p_pred.max() > 1.1) else p_pred.clamp(0, 1)
        p_tgt = torch.sigmoid(p_tgt) if (p_tgt.min() < -0.1 or p_tgt.max() > 1.1) else p_tgt.clamp(0, 1)
        B, _, H, W = p_pred.shape

        ds = max(1, self.downsample)
        if ds > 1 and min(H, W) >= ds:
            p_pred_s = F.avg_pool2d(p_pred, kernel_size=ds, stride=ds)
            with torch.no_grad():
                p_tgt_s = F.avg_pool2d(p_tgt, kernel_size=ds, stride=ds)
        else:
            p_pred_s, p_tgt_s = p_pred, p_tgt
            ds = 1
        with torch.no_grad():
            p_tgt_s = self._maybe_smooth_target(p_tgt_s).clamp(0, 1)

        _, _, Hs, Ws = p_pred_s.shape
        device, dtype = p_pred_s.device, p_pred_s.dtype
        taus = torch.linspace(self.tau_min, self.tau_max, self.M, device=device, dtype=dtype)
        sig = max(1e-6, self.sigma)

        def _compute_chi_all(p_in: torch.Tensor, need_grad: bool) -> torch.Tensor:
            chunksz = self.chunk_size if self.chunk_size and self.chunk_size > 0 else self.M
            outs = []
            for s in range(0, self.M, chunksz):
                e = min(self.M, s + chunksz)
                tau_c = taus[s:e].view(e - s, 1, 1, 1, 1)
                m = torch.sigmoid((p_in.unsqueeze(0) - tau_c) / sig)
                mf = m.reshape((e - s) * B, 1, Hs, Ws)
                chi = self._chi_from_mask(mf).view(e - s, B).transpose(0, 1)
                outs.append(chi if need_grad else chi.detach())
            return torch.cat(outs, dim=1)

        chi_p = _compute_chi_all(p_pred_s, need_grad=True)
        with torch.no_grad():
            chi_t = _compute_chi_all(p_tgt_s, need_grad=False)

        scale = math.sqrt(float(Hs * Ws)) if self.norm == "sqrt" else (float(Hs * Ws) if self.norm == "area" else 1.0)
        scale = max(scale, 1.0)
        chi_p = chi_p / scale
        chi_t = chi_t / scale
        dchi_p = chi_p[:, 1:] - chi_p[:, :-1]
        dchi_t = chi_t[:, 1:] - chi_t[:, :-1]
        birth_p = F.relu(dchi_p)
        death_p = F.relu(-dchi_p)
        birth_t = F.relu(dchi_t)
        death_t = F.relu(-dchi_t)

        with torch.no_grad():
            u_t = (p_tgt_s * (1.0 - p_tgt_s)).mean(dim=(1, 2, 3))
            conf_t = (1.0 - u_t / 0.25).clamp(0.0, 1.0)
        u_p = (p_pred * (1.0 - p_pred)).mean(dim=(1, 2, 3))
        conf_p = (1.0 - u_p / 0.25).clamp(0.0, 1.0)

        with torch.no_grad():
            if self.importance_mix == "tgt":
                crit = dchi_t.abs()
            elif self.importance_mix == "mix":
                crit = conf_t[:, None] * dchi_t.abs() + (1.0 - conf_t[:, None]) * dchi_p.detach().abs()
            else:
                use_mix = conf_t.mean() < self.conf_thr_auto_mix
                crit = conf_t[:, None] * dchi_t.abs() + (1.0 - conf_t[:, None]) * dchi_p.detach().abs() if bool(use_mix) else dchi_t.abs()
            crit_n = crit / (crit.mean(dim=1, keepdim=True).clamp_min(1e-6))
            temp_eff = max(1e-6, float(self.weight_temp)) * (1.0 + float(self._temp_warmup_factor - 1.0) * float((1.0 - warm_coef.mean()).clamp(0.0, 1.0).item()))
            logits_w_pre_mask = crit_n / max(1e-6, temp_eff)
            logits_w = logits_w_pre_mask.clone()
            idx = None
            if self.max_events > 0 and self.max_events < logits_w.shape[1]:
                idx = torch.topk(logits_w, k=self.max_events, dim=1).indices
                mask = torch.full_like(logits_w, -1e9)
                mask.scatter_(1, idx, 0.0)
                logits_w = logits_w + mask
            w_tau = torch.softmax(logits_w, dim=1)
            w_ent = (-(w_tau.clamp_min(1e-12) * w_tau.clamp_min(1e-12).log()).sum(dim=1) / math.log(float(w_tau.shape[1]) + 1e-12)).mean()

        if self.mode == "match":
            mass_all = self._huber(birth_p - birth_t) + self._huber(death_p - death_t)
        elif self.mode == "simplicity":
            mass_all = self._huber(birth_p) + self._huber(death_p)
        else:
            mass_all = self._huber(F.relu(birth_p - birth_t)) + self._huber(F.relu(death_p - death_t))
            with torch.no_grad():
                unc = (1.0 - conf_t).clamp(0.0, 1.0)
            if self.simp_weight_when_uncertain > 0:
                mass_all = mass_all + (self.simp_weight_when_uncertain * unc[:, None]) * (self._huber(birth_p) + self._huber(death_p))

        self._ensure_heights(device, Hs, Ws, dtype)
        heights = self._heights
        eps = 1e-6
        p_pred_hw = p_pred_s[:, 0]
        with torch.no_grad():
            p_tgt_hw = p_tgt_s[:, 0]

        if self.max_events > 0 and self.max_events < (self.M - 1):
            if idx is None:
                with torch.no_grad():
                    if self.importance_mix == "tgt":
                        score = dchi_t.abs()
                    elif self.importance_mix == "mix":
                        score = conf_t[:, None] * dchi_t.abs() + (1.0 - conf_t[:, None]) * dchi_p.detach().abs()
                    else:
                        use_mix = conf_t.mean() < self.conf_thr_auto_mix
                        score = (conf_t[:, None] * dchi_t.abs() + (1.0 - conf_t[:, None]) * dchi_p.detach().abs()) if bool(use_mix) else dchi_t.abs()
                    score_n = score / (score.mean(dim=1, keepdim=True).clamp_min(1e-6))
                    idx = torch.topk(score_n, k=self.max_events, dim=1).indices
            w_ev = w_tau.gather(1, idx)
            w_ev = w_ev / w_ev.sum(dim=1, keepdim=True).clamp_min(1e-6)
            mass_ev = mass_all.gather(1, idx)
            mass_cost = (mass_ev * w_ev).sum(dim=1)
            tau_i = taus[idx]
            tau_j = taus[(idx + 1).clamp_max(self.M - 1)]
            band_p = torch.sigmoid((p_pred_hw[:, None] - tau_i[:, :, None, None]) / sig) - torch.sigmoid((p_pred_hw[:, None] - tau_j[:, :, None, None]) / sig)
            band_p = band_p.clamp_min(0.0)
            with torch.no_grad():
                band_t = torch.sigmoid((p_tgt_hw[:, None] - tau_i[:, :, None, None]) / sig) - torch.sigmoid((p_tgt_hw[:, None] - tau_j[:, :, None, None]) / sig)
                band_t = band_t.clamp_min(0.0)
            denom_p = band_p.sum(dim=(2, 3)).clamp_min(eps)
            denom_t = band_t.sum(dim=(2, 3)).clamp_min(eps)
            hp = (band_p[:, :, None] * heights[None, None]).sum(dim=(3, 4)) / denom_p[:, :, None]
            with torch.no_grad():
                ht = (band_t[:, :, None] * heights[None, None]).sum(dim=(3, 4)) / denom_t[:, :, None]
            pos = (hp - ht).abs().pow(self.p).mean(dim=2)
            pos_cost = (pos * w_ev).sum(dim=1)
            k_used = int(self.max_events)
        else:
            tau_i = taus[:-1].view(1, self.M - 1, 1, 1)
            tau_j = taus[1:].view(1, self.M - 1, 1, 1)
            band_p = torch.sigmoid((p_pred_hw[:, None] - tau_i) / sig) - torch.sigmoid((p_pred_hw[:, None] - tau_j) / sig)
            band_p = band_p.clamp_min(0.0)
            with torch.no_grad():
                band_t = torch.sigmoid((p_tgt_hw[:, None] - tau_i) / sig) - torch.sigmoid((p_tgt_hw[:, None] - tau_j) / sig)
                band_t = band_t.clamp_min(0.0)
            denom_p = band_p.sum(dim=(2, 3)).clamp_min(eps)
            denom_t = band_t.sum(dim=(2, 3)).clamp_min(eps)
            hp = (band_p[:, :, None] * heights[None, None]).sum(dim=(3, 4)) / denom_p[:, :, None]
            with torch.no_grad():
                ht = (band_t[:, :, None] * heights[None, None]).sum(dim=(3, 4)) / denom_t[:, :, None]
            pos = (hp - ht).abs().pow(self.p).mean(dim=2)
            pos_cost = (pos * w_tau).sum(dim=1)
            mass_cost = (mass_all * w_tau).sum(dim=1)
            k_used = int(self.M - 1)

        with torch.no_grad():
            pos_scale = torch.where(conf_t < self.conf_thr_auto_mix, torch.full_like(conf_t, self.pos_weight_when_uncertain), torch.ones_like(conf_t))
            pos_anneal = (0.25 + 0.75 * conf_p).clamp(0.25, 1.0)
            pos_anneal = pos_anneal * (0.2 + 0.8 * warm_coef)
        pos_cost = pos_cost * pos_scale * pos_anneal

        if self.use_uncert_gate:
            with torch.no_grad():
                conf_mix = 0.5 * conf_t + 0.5 * conf_p.detach()
            gate = 0.1 + 0.9 * conf_mix
            if self.gate_detach:
                gate = gate.detach()
        else:
            gate = p_pred.new_ones((B,))

        loss_batch = (self.lambda_mass * mass_cost + self.lambda_pos * pos_cost) * gate
        loss_batch = loss_batch * warm_coef
        loss = loss_batch.mean()
        parts = {
            "ph/loss": loss.detach(),
            "ph/mass": mass_cost.mean().detach(),
            "ph/pos": pos_cost.mean().detach(),
            "ph/gate": gate.mean().detach(),
            "ph/conf_t": conf_t.mean().detach(),
            "ph/conf_p": conf_p.mean().detach(),
            "ph/w_entropy": w_ent.detach() if isinstance(w_ent, torch.Tensor) else torch.tensor(float(w_ent), device=device),
            "ph/warm": warm_coef.mean().detach(),
            "ph/temp_eff": torch.tensor(float(temp_eff), device=device),
            "ph/ds": torch.tensor(float(ds), device=device),
            "ph/M": torch.tensor(float(self.M), device=device),
            "ph/k": torch.tensor(float(k_used), device=device),
            "ph/mode": torch.tensor(0.0 if self.mode == "match" else (1.0 if self.mode == "cap" else 2.0), device=device),
            "ph/skip": torch.tensor(0.0, device=device),
        }
        return loss, parts
