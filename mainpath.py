"""
Mainpath gate objective (paper Eq. 16):

  L_mainpath = sum_l ( -<g_out, g_new> / (||g_out||^2 + eps) + lambda * alpha_l^2 )

with g_new = (1 - alpha) g_out + alpha g_in; g_out/g_in detached for the gate step.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from gradient_modulation import GRAD_EPS


def compute_mainpath_gate_loss(
    blend_stats: Sequence[Dict[str, torch.Tensor]],
    *,
    gate_lambda: float,
    grad_eps: float = GRAD_EPS,
) -> torch.Tensor:
    if not blend_stats:
        return torch.zeros((), device=torch.device("cpu"))
    device = None
    loss = None
    for item in blend_stats:
        alpha = item.get("alpha")
        g_outer = item.get("g_outer")
        g_inner = item.get("g_inner")
        if alpha is None or g_outer is None or g_inner is None:
            continue
        if device is None:
            device = alpha.device
        go = g_outer.reshape(-1).detach()
        gi = g_inner.reshape(-1).detach()
        a = alpha.reshape(())
        g_new = (1.0 - a) * go + a * gi
        denom = go.norm().pow(2) + float(grad_eps)
        term = -torch.dot(go, g_new) / denom + float(gate_lambda) * (a * a)
        loss = term if loss is None else loss + term
    if loss is None:
        return torch.zeros((), device=device or torch.device("cpu"))
    return loss
