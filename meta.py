import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from typing import Dict, List
from copy import deepcopy
from sklearn.metrics import average_precision_score

from learner import Classifier
from gradient_modulation import (
    GradientBlender,
    average_inner_grads,
    build_trainable_layer_groups,
    count_blender_layers,
    modulate_meta_gradients,
    needs_blend_stats,
    needs_blender,
    normalize_modulation_version,
    resolve_blend_layers,
    resolve_blend_mode,
    uses_inner_grad_cache,
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def asymmetric_loss_binary(logits, targets, gamma_pos=0.2, gamma_neg=4, alpha_pos=2.0, alpha_neg=1.0, margin=0.01):
    logits_pos = logits[:, 1]
    probs = torch.sigmoid(logits_pos)
    targets = targets.float()

    pos_inds = targets == 1
    neg_inds = targets == 0

    loss_pos = -alpha_pos * (1 - probs[pos_inds]) ** gamma_pos * torch.log(probs[pos_inds] + 1e-8)
    pm = torch.clamp(probs[neg_inds] - margin, min=0.0)
    loss_neg = -alpha_neg * (pm ** gamma_neg) * torch.log(1 - pm + 1e-8)
    return torch.cat([loss_pos, loss_neg]).mean()


def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')
    l2_reg = torch.norm(logits, p=2)

    _, y_hat = input_cpu.max(1)
    acc_val = y_hat.eq(target_cpu).float().mean()

    loss_val = asymmetric_loss_binary(
        input_cpu.to(logits.device), y_t.to(logits.device),
        gamma_pos=0, gamma_neg=4, alpha_pos=1, alpha_neg=1, margin=0.01,
    )
    total_loss = loss_val + 0.01 * l2_reg

    y_pred_probs = F.softmax(logits, dim=1)
    y_true = target_cpu.cpu().numpy()
    y_pred_probs_1d = y_pred_probs[:, 1].detach().cpu().numpy()

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        auprc = 0.0
        recall_at_k = 0.0
    elif np.isnan(y_pred_probs_1d).any() or np.isinf(y_pred_probs_1d).any():
        auprc = 0.0
        recall_at_k = 0.0
    else:
        if len(np.unique(y_true)) > 1:
            try:
                auprc = average_precision_score(y_true, y_pred_probs_1d)
                if np.isnan(auprc) or np.isinf(auprc):
                    auprc = 0.0
            except ValueError:
                auprc = 0.0
        else:
            auprc = 0.0

        k = int(y_true.sum())
        if k > 0:
            try:
                top_k_indices = np.argsort(y_pred_probs_1d)[-k:]
                recall_at_k = y_true[top_k_indices].sum() / k
                if np.isnan(recall_at_k) or np.isinf(recall_at_k):
                    recall_at_k = 0.0
            except Exception:
                recall_at_k = 0.0
        else:
            recall_at_k = 0.0

    return total_loss, acc_val, auprc, recall_at_k


def proto_loss_qry(logits, y_t):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')
    l2_reg = torch.norm(logits, p=2)

    _, y_hat = input_cpu.max(1)
    acc_val = y_hat.eq(target_cpu).float().mean()

    loss_val = asymmetric_loss_binary(
        input_cpu.to(logits.device), y_t.to(logits.device),
        gamma_pos=0, gamma_neg=4, alpha_pos=1, alpha_neg=1, margin=0.01,
    )
    total_loss = loss_val + 0.01 * l2_reg

    y_pred_probs = F.softmax(logits, dim=1)
    y_true = target_cpu.cpu().numpy()
    y_pred_probs_1d = y_pred_probs[:, 1].detach().cpu().numpy()

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        auprc = 0.0
        recall_at_k = 0.0
    elif np.isnan(y_pred_probs_1d).any() or np.isinf(y_pred_probs_1d).any():
        auprc = 0.0
        recall_at_k = 0.0
    else:
        if len(np.unique(y_true)) > 1:
            try:
                auprc = average_precision_score(y_true, y_pred_probs_1d)
                if np.isnan(auprc) or np.isinf(auprc):
                    auprc = 0.0
            except ValueError:
                auprc = 0.0
        else:
            auprc = 0.0

        k = int(y_true.sum())
        if k > 0:
            try:
                top_k_indices = np.argsort(y_pred_probs_1d)[-k:]
                recall_at_k = y_true[top_k_indices].sum() / k
                if np.isnan(recall_at_k) or np.isinf(recall_at_k):
                    recall_at_k = 0.0
            except Exception:
                recall_at_k = 0.0
        else:
            recall_at_k = 0.0

    return total_loss, acc_val, auprc, recall_at_k


class Meta(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        raw_modulation_version = getattr(args, 'modulation_version', 'LCAM')
        self.modulation_version_label = raw_modulation_version
        self.modulation_version = normalize_modulation_version(raw_modulation_version)
        self.blend_mode = resolve_blend_mode(
            raw_modulation_version,
            getattr(args, 'blend_mode', 'original'),
        )
        self.modulation_aux_weight = getattr(args, 'modulation_aux_weight', 0.01)
        self.aux_lambda = float(getattr(args, 'aux_lambda', 0.01))
        self.gate_objective = str(getattr(args, 'gate_objective', 'mainpath')).strip().lower()
        self.blend_layers = resolve_blend_layers(getattr(args, 'blend_layers', 'all'))
        self.current_epoch = 0
        self.meta_step = 0

        self.net = Classifier(config).to(device)

        freeze_gnn = getattr(args, 'freeze_gnn', True)
        if freeze_gnn:
            self._freeze_gnn_parameters()
        self.freeze_gnn = freeze_gnn

        num_blender_layers = count_blender_layers(self.net, self.blend_layers)
        self.gradient_blender = None
        if needs_blender(self.modulation_version) and num_blender_layers > 0:
            alpha_bias_init = float(getattr(args, 'alpha_bias_init', 0.0))
            self.gradient_blender = GradientBlender(
                num_blender_layers,
                alpha_bias_init=alpha_bias_init,
            ).to(device)

        self._blend_epoch_alpha = {}
        self._blend_epoch_d = {}
        self._blend_epoch_inner_norm = {}
        self._blend_epoch_outer_norm = {}
        self._blend_epoch_cosine = {}

        optim_params = [p for p in self.net.parameters() if p.requires_grad]
        if self.gradient_blender is not None:
            optim_params += list(self.gradient_blender.parameters())
        self.meta_optim = optim.Adam(optim_params, lr=self.meta_lr, betas=(0.9, 0.999), weight_decay=1e-5)
        self.max_grad_norm = 1.0

    def _should_cache_inner_grad(self):
        return uses_inner_grad_cache(self.modulation_version)

    def set_training_context(self, epoch: int, meta_step: int):
        self.current_epoch = int(epoch)
        self.meta_step = int(meta_step)

    def reset_blend_epoch_stats(self):
        if not needs_blend_stats(self.modulation_version):
            return
        layer_groups = build_trainable_layer_groups(self.net)
        self._blend_epoch_alpha = {g.layer_name: [] for g in layer_groups}
        self._blend_epoch_d = {g.layer_name: [] for g in layer_groups}
        self._blend_epoch_inner_norm = {g.layer_name: [] for g in layer_groups}
        self._blend_epoch_outer_norm = {g.layer_name: [] for g in layer_groups}
        self._blend_epoch_cosine = {g.layer_name: [] for g in layer_groups}

    def record_blend_stats(self, blend_diagnostics):
        if not blend_diagnostics:
            return
        for item in blend_diagnostics:
            layer_name = item['layer_name']
            stores = [
                (self._blend_epoch_alpha, float(item['alpha'].detach().item())),
                (self._blend_epoch_inner_norm, float(item['inner_norm'].detach().item())),
                (self._blend_epoch_outer_norm, float(item['outer_norm'].detach().item())),
                (self._blend_epoch_cosine, float(item['cosine'].detach().item())),
            ]
            if needs_blender(self.modulation_version):
                stores.append((self._blend_epoch_d, float(item['d'].detach().item())))
            for store, val in stores:
                store.setdefault(layer_name, []).append(val)

    def format_blend_epoch_stats(self) -> str:
        if not needs_blend_stats(self.modulation_version):
            return ''
        lines = []
        all_inner, all_outer, all_alpha, all_cosine = [], [], [], []
        for layer_name in sorted(self._blend_epoch_alpha.keys()):
            alphas = self._blend_epoch_alpha.get(layer_name, [])
            ds = self._blend_epoch_d.get(layer_name, [])
            inner_norms = self._blend_epoch_inner_norm.get(layer_name, [])
            outer_norms = self._blend_epoch_outer_norm.get(layer_name, [])
            cosines = self._blend_epoch_cosine.get(layer_name, [])
            if not alphas:
                continue
            mean_inner = np.mean(inner_norms)
            mean_outer = np.mean(outer_norms)
            mean_alpha = np.mean(alphas)
            mean_cosine = np.mean(cosines)
            all_inner.append(mean_inner)
            all_outer.append(mean_outer)
            all_alpha.append(mean_alpha)
            all_cosine.append(mean_cosine)
            lines.append(
                f'  {layer_name}: mean_alpha={mean_alpha:.4f}, mean_conflict_D={np.mean(ds):.4f}, '
                f'mean_inner_norm={mean_inner:.4f}, mean_outer_norm={mean_outer:.4f}, '
                f'mean_cosine={mean_cosine:.4f}'
            )
        if all_inner:
            lines.append(
                f'  [all layers] mean_inner_norm={np.mean(all_inner):.4f}, '
                f'mean_outer_norm={np.mean(all_outer):.4f}, mean_alpha={np.mean(all_alpha):.4f}, '
                f'mean_cosine={np.mean(all_cosine):.4f}, blend_mode={self.blend_mode}'
            )
        if not lines:
            return ''
        return f'{self.modulation_version_label} stats:\n' + '\n'.join(lines)

    def _freeze_gnn_parameters(self):
        gnn_layer_names = {'GraphConv'}
        param_idx = 0
        for layer_name, _ in self.net.config:
            if layer_name in gnn_layer_names:
                if param_idx < len(self.net.vars):
                    self.net.vars[param_idx].requires_grad = False
                if param_idx + 1 < len(self.net.vars):
                    self.net.vars[param_idx + 1].requires_grad = False
                param_idx += 2
            elif layer_name == 'Linear':
                if param_idx < len(self.net.vars):
                    self.net.vars[param_idx].requires_grad = True
                if param_idx + 1 < len(self.net.vars):
                    self.net.vars[param_idx + 1].requires_grad = True
                param_idx += 2

        for conv in self.net.graph_conv:
            if hasattr(conv, 'beta'):
                conv.beta.requires_grad = False

    def forward_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat):
        task_num = len(x_spt)
        losses_s = [0 for _ in range(self.update_step)]
        accs_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        auprcs = [0 for _ in range(self.update_step + 1)]
        recall_at_ks = [0 for _ in range(self.update_step + 1)]
        inner_grads_per_task = []

        for i in range(task_num):
            last_inner_grad = None
            feat_spt = torch.Tensor(np.vstack(([feat[0][np.array(x)] for x in n_spt[i]]))).to(device)
            feat_qry = torch.Tensor(np.vstack(([feat[0][np.array(x)] for x in n_qry[i]]))).to(device)

            logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, vars=None)
            loss, acc_val, _, _ = proto_loss_spt(logits, y_spt[i], self.k_spt)
            losses_s[0] += loss
            accs_s[0] += acc_val

            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            grad = torch.autograd.grad(loss, trainable_params, allow_unused=True, create_graph=True)
            grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, trainable_params)]
            if self._should_cache_inner_grad():
                last_inner_grad = [
                    g.detach().clone() if g is not None else torch.zeros_like(p)
                    for g, p in zip(grad, trainable_params)
                ]

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, trainable_params)))
            all_params = list(self.net.parameters())
            fast_weights_full = []
            trainable_idx = 0
            for param in all_params:
                if param.requires_grad:
                    fast_weights_full.append(fast_weights[trainable_idx])
                    trainable_idx += 1
                else:
                    fast_weights_full.append(param)
            fast_weights = fast_weights_full

            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, self.net.parameters())
                loss_q, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry[i])
                losses_q[0] += loss_q
                corrects[0] += acc_q
                auprcs[0] += auprc_q
                recall_at_ks[0] += recall_at_k_q

            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                loss_q, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry[i])
                losses_q[1] += loss_q
                corrects[1] += acc_q
                auprcs[1] += auprc_q
                recall_at_ks[1] += recall_at_k_q

            for k in range(1, self.update_step):
                logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, fast_weights)
                loss, acc_val, _, _ = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                accs_s[k] += acc_val

                trainable_fast_weights = [w for w, p in zip(fast_weights, all_params) if p.requires_grad]
                grad = torch.autograd.grad(loss, trainable_fast_weights, retain_graph=True, allow_unused=True)
                grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, trainable_fast_weights)]
                if self._should_cache_inner_grad():
                    last_inner_grad = [
                        g.detach().clone() if g is not None else torch.zeros_like(w)
                        for g, w in zip(grad, trainable_fast_weights)
                    ]

                updated_trainable = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, trainable_fast_weights)))
                trainable_idx = 0
                for i_param, param in enumerate(all_params):
                    if param.requires_grad:
                        fast_weights[i_param] = updated_trainable[trainable_idx]
                        trainable_idx += 1

                logits_q, _ = self.net(
                    x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights,
                )
                loss_q, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                corrects[k + 1] += acc_q
                auprcs[k + 1] += auprc_q
                recall_at_ks[k + 1] += recall_at_k_q

            if self._should_cache_inner_grad() and last_inner_grad is not None:
                inner_grads_per_task.append(last_inner_grad)

        loss_q = losses_q[-1] / task_num
        if not torch.isnan(loss_q):
            self.meta_optim.zero_grad()

            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            inner_grads_avg = None
            if self._should_cache_inner_grad() and inner_grads_per_task:
                inner_grads_avg = average_inner_grads(inner_grads_per_task, trainable_params)

            loss_q.backward(retain_graph=False)

            if uses_inner_grad_cache(self.modulation_version) and inner_grads_avg is not None:
                new_grads, blend_diagnostics = modulate_meta_gradients(
                    self.net,
                    trainable_params,
                    inner_grads_avg,
                    version=self.modulation_version,
                    blender=self.gradient_blender,
                    blend_mode=self.blend_mode,
                    blend_layers=self.blend_layers,
                )
                for param, new_grad in zip(trainable_params, new_grads):
                    param.grad = new_grad
                self.record_blend_stats(blend_diagnostics)
                if self.gradient_blender is not None and blend_diagnostics:
                    if self.gate_objective == 'mainpath':
                        from mainpath import compute_mainpath_gate_loss

                        blend_aux = compute_mainpath_gate_loss(
                            blend_diagnostics,
                            gate_lambda=self.aux_lambda,
                        )
                    else:
                        blend_aux = self.modulation_aux_weight * self.gradient_blender.auxiliary_loss(
                            blend_diagnostics,
                            aux_lambda=self.aux_lambda,
                        )
                    if blend_aux.requires_grad:
                        blend_aux.backward()

            clip_params = list(self.net.parameters())
            if self.gradient_blender is not None:
                clip_params += list(self.gradient_blender.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=self.max_grad_norm)
            self.meta_optim.step()

        accs = np.array(corrects) / task_num
        auprcs = np.array(auprcs) / task_num
        recall_at_ks = np.array(recall_at_ks) / task_num
        return accs, loss_q, auprcs, recall_at_ks

    def finetunning_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat):
        corrects = [0 for _ in range(self.update_step_test + 1)]
        auprcs = [0 for _ in range(self.update_step_test + 1)]
        recall_at_ks = [0 for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        n_spt = n_spt[0]
        n_qry = n_qry[0]
        feat_spt = torch.Tensor(np.vstack(([feat[0][np.array(x)] for x in n_spt]))).to(device)
        feat_qry = torch.Tensor(np.vstack(([feat[0][np.array(x)] for x in n_qry]))).to(device)

        logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt)
        loss, _, _, _ = proto_loss_spt(logits, y_spt, self.k_spt)
        params = list(net.parameters())
        trainable_params = [p for p in params if p.requires_grad]
        if trainable_params:
            grad = torch.autograd.grad(loss, trainable_params, retain_graph=True, allow_unused=True)
            fast_weights = []
            trainable_idx = 0
            for p in params:
                if p.requires_grad and trainable_idx < len(grad):
                    g = grad[trainable_idx]
                    fast_weights.append(p - self.update_lr * g if g is not None else p)
                    trainable_idx += 1
                else:
                    fast_weights.append(p)
        else:
            fast_weights = params

        with torch.no_grad():
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, net.parameters())
            _, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry)
            corrects[0] += acc_q
            auprcs[0] += auprc_q
            recall_at_ks[0] += recall_at_k_q

        with torch.no_grad():
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            _, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry)
            corrects[1] += acc_q
            auprcs[1] += auprc_q
            recall_at_ks[1] += recall_at_k_q

        for k in range(1, self.update_step_test):
            logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt, fast_weights)
            loss, _, _, _ = proto_loss_spt(logits, y_spt, self.k_spt)
            trainable_fast_weights = [w for w in fast_weights if w.requires_grad]
            if trainable_fast_weights:
                grad = torch.autograd.grad(loss, trainable_fast_weights, retain_graph=True, allow_unused=True)
                updated_fast_weights = []
                trainable_idx = 0
                for w in fast_weights:
                    if w.requires_grad and trainable_idx < len(grad):
                        g = grad[trainable_idx]
                        updated_fast_weights.append(w - self.update_lr * g if g is not None else w)
                        trainable_idx += 1
                    else:
                        updated_fast_weights.append(w)
                fast_weights = updated_fast_weights

            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            _, acc_q, auprc_q, recall_at_k_q = proto_loss_qry(logits_q, y_qry)
            corrects[k + 1] += acc_q
            auprcs[k + 1] += auprc_q
            recall_at_ks[k + 1] += recall_at_k_q

        del net
        return np.array(corrects), np.array(auprcs), np.array(recall_at_ks)

    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat):
        return self.forward_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat)

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat):
        return self.finetunning_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat)
