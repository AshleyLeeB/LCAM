from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

GRAD_EPS = 1e-8

GNN_LAYER_NAMES = {'GraphConv'}

LCAM_ALIASES = {
    'LCAM', 'lcam',
    'V3_blend', 'v3_blend', 'V3_blending', 'v3_blending',
}


@dataclass
class LayerGroup:
    layer_idx: int
    layer_name: str
    param_indices: List[int]
    trainable: bool


def normalize_modulation_version(version: str) -> str:
    if version in LCAM_ALIASES:
        return 'LCAM'
    raise ValueError(
        f'Unsupported modulation_version: {version!r}; expected LCAM.'
    )


def resolve_blend_mode(_version: str, blend_mode: str = 'original') -> str:
    if blend_mode not in ('original', 'normalized'):
        raise ValueError(f'Invalid blend_mode: {blend_mode!r}; expected original or normalized')
    return blend_mode


BLEND_LAYER_MODES = ('all', 'linear')


def resolve_blend_layers(blend_layers: str = 'all') -> str:
    mode = (blend_layers or 'all').strip().lower()
    if mode in ('*', 'full', 'all_layers'):
        mode = 'all'
    if mode in ('linear_only', 'lin', 'classifier', 'fc'):
        mode = 'linear'
    if mode not in BLEND_LAYER_MODES:
        raise ValueError(
            f'Invalid blend_layers: {blend_layers!r}; expected one of {BLEND_LAYER_MODES}'
        )
    return mode


def layer_allowed_for_blend(layer_name: str, blend_layers: str = 'all') -> bool:
    if resolve_blend_layers(blend_layers) == 'all':
        return True
    return layer_name == 'Linear'


class GradientBlender(nn.Module):
    def __init__(self, num_layers: int, alpha_bias_init: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.alpha_weight = nn.Parameter(torch.zeros(num_layers))
        self.alpha_bias = nn.Parameter(torch.full((num_layers,), float(alpha_bias_init)))

    def compute_alpha(self, layer_id: int, d_val: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.alpha_weight[layer_id] * d_val.detach() + self.alpha_bias[layer_id]
        )

    def blend_gradients(
        self,
        g_outer: torch.Tensor,
        g_inner: torch.Tensor,
        layer_id: int,
        d_val: torch.Tensor,
        blend_mode: str = 'original',
        eps: float = GRAD_EPS,
        detach_alpha: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        alpha = self.compute_alpha(layer_id, d_val)
        alpha_mix = alpha.detach() if detach_alpha else alpha
        inner_norm = torch.norm(g_inner)
        outer_norm = torch.norm(g_outer)
        if inner_norm.item() >= eps and outer_norm.item() >= eps:
            cosine = torch.dot(g_inner, g_outer) / (inner_norm * outer_norm + eps)
        else:
            cosine = torch.zeros((), device=g_inner.device, dtype=g_inner.dtype)

        if float(alpha_mix.item()) == 0.0:
            g_blend = g_outer
        elif blend_mode == 'normalized':
            g_inner_unit = g_inner / (inner_norm + eps)
            g_outer_unit = g_outer / (outer_norm + eps)
            g_dir = (1.0 - alpha_mix) * g_outer_unit + alpha_mix * g_inner_unit
            g_blend = outer_norm * g_dir
        else:
            g_blend = (1.0 - alpha_mix) * g_outer + alpha_mix * g_inner

        stats = {
            'alpha': alpha,
            'd': d_val.detach(),
            'inner_norm': inner_norm.detach(),
            'outer_norm': outer_norm.detach(),
            'cosine': cosine.detach(),
            'g_outer': g_outer.detach(),
            'g_inner': g_inner.detach(),
            'g_blend': g_blend,
        }
        return g_blend, stats

    def auxiliary_loss(
        self,
        blend_stats: Sequence[Dict[str, torch.Tensor]],
        aux_lambda: float = 0.01,
        aux_fo_weight: float = 1.0,
    ) -> torch.Tensor:
        if not blend_stats:
            return torch.zeros((), device=self.alpha_weight.device)
        loss = torch.zeros((), device=self.alpha_weight.device)
        fo_w = float(aux_fo_weight)
        lam = float(aux_lambda)
        for item in blend_stats:
            alpha = item['alpha']
            loss = loss + lam * (alpha * alpha)
            if fo_w <= 0:
                continue
            g_outer = item.get('g_outer')
            g_blend = item.get('g_blend')
            if g_outer is None or g_blend is None or not g_blend.requires_grad:
                continue
            go = g_outer.reshape(-1)
            gb = g_blend.reshape(-1)
            denom = go.norm().pow(2) + GRAD_EPS
            loss = loss + fo_w * (-torch.dot(go, gb) / denom)
        return loss


def build_trainable_layer_groups(net) -> List[LayerGroup]:
    groups = []
    param_idx = 0
    for layer_idx, (layer_name, _) in enumerate(net.config):
        if layer_name in GNN_LAYER_NAMES or layer_name == 'Linear':
            indices = [param_idx, param_idx + 1]
            param_idx += 2
        else:
            continue

        if all(net.vars[i].requires_grad for i in indices):
            groups.append(
                LayerGroup(
                    layer_idx=layer_idx,
                    layer_name=layer_name,
                    param_indices=indices,
                    trainable=True,
                )
            )
    return groups


def count_blender_layers(net, blend_layers: str = 'all') -> int:
    mode = resolve_blend_layers(blend_layers)
    groups = build_trainable_layer_groups(net)
    return sum(1 for g in groups if layer_allowed_for_blend(g.layer_name, mode))


def conflict_d(g_inner: torch.Tensor, g_outer: torch.Tensor, eps: float = GRAD_EPS) -> torch.Tensor:
    norm_in = torch.norm(g_inner)
    norm_out = torch.norm(g_outer)
    if norm_in.item() < eps or norm_out.item() < eps:
        return torch.zeros((), device=g_inner.device, dtype=g_inner.dtype)
    cosine_sim = torch.dot(g_inner, g_outer) / (norm_in * norm_out + eps)
    return 1.0 - cosine_sim


def _var_to_trainable_idx(net) -> Dict[int, int]:
    mapping = {}
    trainable_idx = 0
    for var_idx, var in enumerate(net.vars):
        if var.requires_grad:
            mapping[var_idx] = trainable_idx
            trainable_idx += 1
    return mapping


def _modulate_group(
    g_outer: torch.Tensor,
    g_inner: torch.Tensor,
    blender: Optional[GradientBlender],
    layer_id: int,
    layer_name: str,
    blend_mode: str = 'original',
    detach_alpha: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    d_val = conflict_d(g_inner, g_outer)
    if blender is None:
        return g_outer, None
    g_blend, blend_core = blender.blend_gradients(
        g_outer,
        g_inner,
        layer_id,
        d_val,
        blend_mode=blend_mode,
        detach_alpha=detach_alpha,
    )
    return g_blend, {'layer_name': layer_name, 'layer_id': layer_id, **blend_core}


def average_inner_grads(
    inner_grads_per_task: Sequence[Sequence[torch.Tensor]],
    trainable_params: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    if not inner_grads_per_task:
        return [torch.zeros_like(p) for p in trainable_params]
    avg = []
    for p_idx in range(len(trainable_params)):
        task_grads = [task_grads_list[p_idx] for task_grads_list in inner_grads_per_task]
        avg.append(torch.stack(task_grads, dim=0).mean(dim=0))
    return avg


def modulate_meta_gradients(
    net,
    trainable_params: Sequence[torch.nn.Parameter],
    inner_grads_avg: Sequence[torch.Tensor],
    version: str,
    blender: Optional[GradientBlender] = None,
    blend_mode: str = 'original',
    detach_alpha: bool = True,
    blend_layers: str = 'all',
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    normalize_modulation_version(version)
    blend_layers = resolve_blend_layers(blend_layers)
    layer_groups = build_trainable_layer_groups(net)
    var_map = _var_to_trainable_idx(net)

    grouped_outer = []
    grouped_inner = []
    offset_maps: List[List[int]] = []
    active_groups: List[LayerGroup] = []

    for group in layer_groups:
        parts_outer, parts_inner, indices = [], [], []
        for var_idx in group.param_indices:
            t_idx = var_map.get(var_idx)
            if t_idx is None:
                continue
            g_outer = trainable_params[t_idx].grad
            if g_outer is None:
                g_outer = torch.zeros_like(trainable_params[t_idx])
            parts_outer.append(g_outer.reshape(-1))
            parts_inner.append(inner_grads_avg[t_idx].reshape(-1))
            indices.append(t_idx)

        if not parts_outer:
            continue

        active_groups.append(group)
        grouped_outer.append(torch.cat(parts_outer))
        grouped_inner.append(torch.cat(parts_inner))
        offset_maps.append(indices)

    new_grads = [torch.zeros_like(p) for p in trainable_params]
    blend_diagnostics: List[Dict[str, torch.Tensor]] = []
    blend_layer_id = 0
    for group, g_outer_g, g_inner_g, indices in zip(
        active_groups, grouped_outer, grouped_inner, offset_maps
    ):
        if not layer_allowed_for_blend(group.layer_name, blend_layers):
            g_mod = g_outer_g
            blend_stat = None
        else:
            g_mod, blend_stat = _modulate_group(
                g_outer_g,
                g_inner_g,
                blender,
                blend_layer_id,
                group.layer_name,
                blend_mode=blend_mode,
                detach_alpha=detach_alpha,
            )
            blend_layer_id += 1
        if blend_stat is not None:
            blend_diagnostics.append(blend_stat)
        offset = 0
        for t_idx in indices:
            numel = trainable_params[t_idx].numel()
            new_grads[t_idx] = g_mod[offset: offset + numel].view_as(trainable_params[t_idx])
            offset += numel

    return new_grads, blend_diagnostics


def needs_blender(version: str) -> bool:
    normalize_modulation_version(version)
    return True


def needs_blend_stats(version: str) -> bool:
    normalize_modulation_version(version)
    return True


def uses_inner_grad_cache(version: str) -> bool:
    normalize_modulation_version(version)
    return True
