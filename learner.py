import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, beta=0.5):
        super().__init__()
        self._activation = activation
        self.device = device
        self.in_proj = nn.Linear(out_feats, out_feats // 3).to(self.device)
        self.out_proj = nn.Linear(out_feats, out_feats // 3).to(self.device)
        self.feat_proj = nn.Linear(out_feats, out_feats - 2 * (out_feats // 3)).to(self.device)
        self.beta = nn.Parameter(
            torch.tensor(float(beta), dtype=torch.float, device=self.device)
        )

    def compute_time_density(self, time_ids):
        hist = torch.histc(time_ids, bins=10, min=0.0, max=1.0) + 1e-8
        time_density = hist / hist.sum()
        time_density = time_density / (torch.max(time_density) + 1e-8)
        bin_indices = (time_ids * 9).long().clamp(0, 9)
        return time_density[bin_indices].unsqueeze(-1)

    def _edge_weight(self, graph):
        time_ids = graph.edata['time_id'].float()
        t_density = self.compute_time_density(time_ids)
        if 'amount' in graph.edata:
            amount_raw = graph.edata['amount'].float()
            if amount_raw.dim() == 1:
                amount_raw = amount_raw.unsqueeze(-1)
            elif amount_raw.dim() > 1 and amount_raw.shape[1] > 1:
                amount_raw = amount_raw[:, 0:1]
            pre = amount_raw + self.beta * t_density
        else:
            pre = self.beta * t_density
        pre_min, pre_max = pre.min(), pre.max()
        pre_01 = (pre - pre_min) / (pre_max - pre_min + 1e-8)
        return pre_01 * 2.0 + 1.0

    def message_func_in(self, edges):
        w = edges.data['edge_weight']
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        elif w.dim() > 1 and w.shape[1] > 1:
            w = w[:, 0:1]
        return {'msg_1': edges.src['h'] * w}

    def message_func_out(self, edges):
        w = edges.data['edge_weight']
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        elif w.dim() > 1 and w.shape[1] > 1:
            w = w[:, 0:1]
        return {'msg_2': edges.dst['h'] * w}

    def reduce_func_in(self, nodes):
        return {'msg_1': nodes.mailbox['msg_1'].sum(dim=1)}

    def reduce_func_out(self, nodes):
        return {'msg_2': nodes.mailbox['msg_2'].sum(dim=1)}

    def forward(self, graph, feat, weight, bias):
        graph = graph.to(self.device)
        feat = feat.to(self.device)
        graph = graph.local_var()
        in_degree = graph.in_degrees().float().clamp(min=1)
        out_degree = graph.out_degrees().float().clamp(min=1)
        norm = torch.pow(in_degree * out_degree, -0.5).view(-1, 1).to(feat.device)
        feat = feat * norm
        graph.edata['edge_weight'] = self._edge_weight(graph)
        feat = torch.matmul(feat, weight)
        graph.ndata['h'] = feat
        graph.update_all(self.message_func_in, self.reduce_func_in)
        h_in = graph.ndata['msg_1']
        graph.update_all(self.message_func_out, self.reduce_func_out)
        h_out = graph.ndata['msg_2']
        rst = torch.cat([self.in_proj(h_in), self.out_proj(h_out), self.feat_proj(feat)], dim=-1)
        rst = rst * norm + bias
        if self._activation is not None:
            rst = self._activation(rst)
        return rst


def _gather_centers(g, h, to_fetch):
    num_nodes_ = g.batch_num_nodes()
    temp = [0] + num_nodes_.tolist()
    offset = torch.cumsum(torch.LongTensor(temp), dim=0)[:-1].to(device)
    return h[to_fetch + offset]


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vars = nn.ParameterList()
        self.graph_conv = []
        self.config = config
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float))

        for name, param in self.config:
            if name == 'Linear':
                w = nn.Parameter(torch.ones(param[1], param[0]))
                init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            elif name == 'GraphConv':
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(
                    GraphConv(param[0], param[1], activation=F.relu, beta=self.beta)
                )
            else:
                raise ValueError(f'Unsupported layer {name!r}; use GraphConv and Linear only.')

    def forward(self, g, to_fetch, features, vars=None, return_embed=False):
        if vars is None:
            vars = self.vars

        idx = 0
        i_gc = 0
        h = features.float().clone().to(device)
        query_embed = None

        for name, _param in self.config:
            if name == 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                h = self.graph_conv[i_gc](g, h, w, b)
                g.ndata['h'] = h
                idx += 2
                i_gc += 1
                if i_gc == len(self.graph_conv):
                    h = _gather_centers(g, h, to_fetch)
            elif name == 'Linear':
                w, b = vars[idx], vars[idx + 1]
                if return_embed:
                    query_embed = h.clone()
                h = F.linear(h.float().clone(), w, b)
                idx += 2

        if return_embed and query_embed is not None:
            return h, query_embed
        return h, h

    def zero_grad(self, vars=None):
        with torch.no_grad():
            for p in (self.vars if vars is None else vars):
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        params = list(self.vars)
        for conv in self.graph_conv:
            params.append(conv.beta)
        return params
