import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import csv
import random
import pickle
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import itertools


def build_adjacency_lists(graph):
    num_nodes = graph.num_nodes()
    adj_in = [[] for _ in range(num_nodes)]
    adj_out = [[] for _ in range(num_nodes)]

    src, dst = graph.edges()
    src = src.numpy()
    dst = dst.numpy()
    for u, v in zip(src, dst):
        adj_out[int(u)].append(int(v))
        adj_in[int(v)].append(int(u))
    return adj_in, adj_out


def meta_collate_fn(samples):
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry = zip(*samples)
    return (
        list(graphs_spt),
        list(labels_spt),
        list(graph_qry),
        list(labels_qry),
        list(center_spt),
        list(center_qry),
        list(nodeidx_spt),
        list(nodeidx_qry),
    )


class FlattenedTaskDataset(Dataset):
    def __init__(self, train_datasets):
        self.train_datasets = train_datasets
        self.task_map = []
        self.rebuild_task_map()

    def rebuild_task_map(self):
        self.task_map = []
        for dataset_idx, dataset in enumerate(self.train_datasets):
            for task_idx in range(len(dataset)):
                self.task_map.append((dataset_idx, task_idx))

    def __getitem__(self, index):
        dataset_idx, task_idx = self.task_map[index]
        return self.train_datasets[dataset_idx][task_idx]

    def __len__(self):
        return len(self.task_map)


class Subgraphs(Dataset):
    def __init__(
        self,
        root,
        mode,
        subgraph2label,
        n_way,
        k_shot,
        k_query,
        batchsz,
        adjs,
        h,
        task_labels,
        max_nodes=100,
        max_samples_0=4500,
        max_samples_1=200,
        support_other_count=1,
        query_other_count=2,
        subgraph_cache=None,
    ):
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.h = h
        self.max_nodes = max_nodes
        self.task_labels = task_labels
        self.max_samples_0 = max_samples_0
        self.max_samples_1 = max_samples_1
        self.support_other_count = support_other_count
        self.query_other_count = query_other_count
        self.mode = mode
        self.subgraph2label = subgraph2label
        dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(os.path.join(root, mode + '.csv'))

        self.G = []
        for i in adjs:
            self.G.append(i)
        self.adj_in, self.adj_out = build_adjacency_lists(self.G[0])
        self.subgraph_cache = subgraph_cache
        self.subgraphs = {}

        self.data = []
        for i, (k, v) in enumerate(dictLabels.items()):
            self.data.append(v)
        self.cls_num = len(self.data)
        self.dictLabels = dictLabels
        self.extract_batch(self.batchsz, dictLabels)

    def loadCSV(self, csvf):
        dictGraphs = {0: []}
        dictGraphsLabels = {0: {}}
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for row in csvreader:
                filename = row[1]
                label = row[2]
                dictGraphs[0].append(filename)
                if label not in dictGraphsLabels[0]:
                    dictGraphsLabels[0][label] = []
                dictGraphsLabels[0][label].append(filename)
                if label not in dictLabels:
                    dictLabels[label] = []
                dictLabels[label].append(filename)
        return dictLabels, dictGraphs, dictGraphsLabels

    def extract_batch(self, batchsz, dictLabels):
        self.all_batches = []

        for _ in range(batchsz):
            support_nodes = []
            query_nodes = []

            if self.mode.startswith('train'):
                label_0 = '0'
                other_labels = [lbl for lbl in self.task_labels if lbl != label_0]

                label_0_samples = np.array(dictLabels[label_0])
                other_samples = {lbl: np.array(dictLabels[lbl]) for lbl in other_labels}

                other_total = len(other_labels)
                support_other_count = self.support_other_count
                query_other_count = self.query_other_count

                support_0_count = self.k_shot - other_total * support_other_count
                query_0_count = self.k_query - other_total * query_other_count

                assert support_0_count + query_0_count <= len(label_0_samples), "Not enough label 0 samples"
                for lbl in other_labels:
                    assert support_other_count + query_other_count <= len(other_samples[lbl]), f"Not enough samples for label {lbl}"

                np.random.shuffle(label_0_samples)
                selected_0 = label_0_samples[:support_0_count + query_0_count]
                selected_support_0 = selected_0[:support_0_count]
                selected_query_0 = selected_0[support_0_count:]

                selected_support_other = []
                selected_query_other = []

                for lbl in other_labels:
                    np.random.shuffle(other_samples[lbl])
                    sel = other_samples[lbl][:support_other_count + query_other_count]
                    selected_support_other.extend(sel[:support_other_count])
                    selected_query_other.extend(sel[support_other_count:])

                support_nodes = np.concatenate([selected_support_0, selected_support_other])
                query_nodes = np.concatenate([selected_query_0, selected_query_other])

                support_nodes = np.random.permutation(support_nodes)
                query_nodes = np.random.permutation(query_nodes)

                support_nodes = [np.array([nid], dtype='<U6') for nid in support_nodes]
                query_nodes = [np.array([nid], dtype='<U6') for nid in query_nodes]

                self.all_batches.append({'support': support_nodes, 'query': query_nodes})
            else:
                label_0_samples = np.array(dictLabels['0'])
                anomaly_label = [lbl for lbl in self.task_labels if lbl != '0'][0] if len([lbl for lbl in self.task_labels if lbl != '0']) > 0 else '1'
                label_6_samples = np.array(dictLabels.get(anomaly_label, []))

                total = len(label_0_samples) + len(label_6_samples)
                if total == 0:
                    raise ValueError(f"No samples found for labels 0 and {anomaly_label}.")

                label_0_samples = np.array(dictLabels['0'])
                label_6_samples = np.array(dictLabels.get(anomaly_label, []))

                np.random.shuffle(label_0_samples)
                np.random.shuffle(label_6_samples)

                label_0_samples = label_0_samples[:min(self.max_samples_0, len(label_0_samples))]
                label_6_samples = label_6_samples[:min(self.max_samples_1, len(label_6_samples))]

                split_idx_0 = len(label_0_samples) // 2
                selected_support_0 = label_0_samples[:split_idx_0]
                selected_query_0 = label_0_samples[split_idx_0:]
                split_idx_6 = len(label_6_samples) // 2
                selected_support_6 = label_6_samples[:split_idx_6]
                selected_query_6 = label_6_samples[split_idx_6:]
                support_nodes = np.concatenate([selected_support_0, selected_support_6])
                query_nodes = np.concatenate([selected_query_0, selected_query_6])
                support_nodes = np.random.permutation(support_nodes)
                query_nodes = np.random.permutation(query_nodes)
                support_nodes = [np.array([nid], dtype='<U6') for nid in support_nodes]
                query_nodes = [np.array([nid], dtype='<U6') for nid in query_nodes]
                self.all_batches.append({'support': support_nodes, 'query': query_nodes})

        return self.all_batches

    def _collect_neighbors(self, center_id):
        i = int(center_id)
        if self.h == 1:
            neighbors = set(self.adj_in[i])
            neighbors.update(self.adj_out[i])
            neighbors.add(i)
            return np.array(sorted(neighbors), dtype=np.int64)

        if self.h == 2:
            f_hop_in = self.adj_in[i]
            f_hop_out = self.adj_out[i]
            neighbors = set(f_hop_in)
            neighbors.update(f_hop_out)
            neighbors.add(i)
            for n in f_hop_in:
                neighbors.update(self.adj_in[n])
                neighbors.update(self.adj_out[n])
            for n in f_hop_out:
                neighbors.update(self.adj_in[n])
                neighbors.update(self.adj_out[n])
            return np.array(sorted(neighbors), dtype=np.int64)

        G = self.G[0]
        if self.h == 3:
            f_hop_in = [n.item() for n in G.in_edges(i)[0]]
            n_l_1 = [[n.item() for n in G.in_edges(x)[0]] for x in f_hop_in]
            n_l_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1]
            n_l_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1]
            n_l_2 = [[n.item() for n in G.out_edges(x)[1]] for x in f_hop_in]
            n_l_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2]
            n_l_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2]
            f_hop_out = [n.item() for n in G.out_edges(i)[1]]
            n_l_3 = [[n.item() for n in G.in_edges(x)[0]] for x in f_hop_out]
            n_l_3_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3]
            n_l_3_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3]
            n_l_4 = [[n.item() for n in G.out_edges(x)[1]] for x in f_hop_out]
            n_l_4_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4]
            n_l_4_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4]
            neighbors = set(
                list(itertools.chain(*n_l_1))
                + list(itertools.chain(*n_l_2))
                + list(itertools.chain(*n_l_3))
                + list(itertools.chain(*n_l_4))
                + f_hop_out
                + f_hop_in
                + [i]
                + list(itertools.chain(*n_l_1_1))
                + list(itertools.chain(*n_l_1_2))
                + list(itertools.chain(*n_l_2_1))
                + list(itertools.chain(*n_l_2_2))
                + list(itertools.chain(*n_l_3_1))
                + list(itertools.chain(*n_l_3_2))
                + list(itertools.chain(*n_l_4_1))
                + list(itertools.chain(*n_l_4_2))
            )
            return np.array(sorted(neighbors), dtype=np.int64)

        f_hop_in = [n.item() for n in G.in_edges(i)[0]]
        n_l_1 = [[n.item() for n in G.in_edges(x)[0]] for x in f_hop_in]
        n_l_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1]
        n_l_1_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1_1]
        n_l_1_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1_1]
        n_l_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1]
        n_l_1_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1_2]
        n_l_1_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_1_2]
        n_l_2 = [[n.item() for n in G.out_edges(x)[1]] for x in f_hop_in]
        n_l_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2]
        n_l_2_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2_1]
        n_l_2_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2_1]
        n_l_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2]
        n_l_2_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2_2]
        n_l_2_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_2_2]
        f_hop_out = [n.item() for n in G.out_edges(i)[1]]
        n_l_3 = [[n.item() for n in G.in_edges(x)[0]] for x in f_hop_out]
        n_l_3_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3]
        n_l_3_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3_1]
        n_l_3_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3_1]
        n_l_3_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3]
        n_l_3_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3_2]
        n_l_3_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_3_2]
        n_l_4 = [[n.item() for n in G.out_edges(x)[1]] for x in f_hop_out]
        n_l_4_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4]
        n_l_4_1_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4_1]
        n_l_4_1_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4_1]
        n_l_4_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4]
        n_l_4_2_1 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4_1]
        n_l_4_2_2 = [[n.item() for n in G.in_edges(x)[0]] for x in n_l_4_1]
        neighbors = set(
            list(itertools.chain(*n_l_1))
            + list(itertools.chain(*n_l_2))
            + list(itertools.chain(*n_l_3))
            + list(itertools.chain(*n_l_4))
            + f_hop_out
            + f_hop_in
            + [i]
            + list(itertools.chain(*n_l_1_1))
            + list(itertools.chain(*n_l_1_2))
            + list(itertools.chain(*n_l_2_1))
            + list(itertools.chain(*n_l_2_2))
            + list(itertools.chain(*n_l_3_1))
            + list(itertools.chain(*n_l_3_2))
            + list(itertools.chain(*n_l_4_1))
            + list(itertools.chain(*n_l_4_2))
            + list(itertools.chain(*n_l_1_1_1))
            + list(itertools.chain(*n_l_1_1_2))
            + list(itertools.chain(*n_l_1_2_1))
            + list(itertools.chain(*n_l_1_2_2))
            + list(itertools.chain(*n_l_2_1_1))
            + list(itertools.chain(*n_l_2_1_2))
            + list(itertools.chain(*n_l_2_2_1))
            + list(itertools.chain(*n_l_2_2_2))
            + list(itertools.chain(*n_l_3_1_1))
            + list(itertools.chain(*n_l_3_1_2))
            + list(itertools.chain(*n_l_3_2_1))
            + list(itertools.chain(*n_l_3_2_2))
            + list(itertools.chain(*n_l_4_1_1))
            + list(itertools.chain(*n_l_4_1_2))
            + list(itertools.chain(*n_l_4_2_1))
            + list(itertools.chain(*n_l_4_2_2))
        )
        return np.array(sorted(neighbors), dtype=np.int64)

    def _limit_neighbors(self, G, center_id, h_hops_neighbor, item):
        i = int(center_id)
        if len(h_hops_neighbor) <= self.max_nodes:
            return h_hops_neighbor

        if isinstance(h_hops_neighbor, np.ndarray):
            neighbor_list = h_hops_neighbor.tolist()
        else:
            neighbor_list = list(h_hops_neighbor)

        f_hop_in = self.adj_in[i]
        f_hop_out = self.adj_out[i]
        direct_neighbors = list(set(f_hop_in + f_hop_out))
        other_nodes = [n for n in neighbor_list if n != i and n not in direct_neighbors]
        rng = np.random.RandomState(int(item) % 2**31)

        num_direct = len(direct_neighbors)
        num_to_sample = self.max_nodes - 1 - num_direct

        if num_direct > self.max_nodes - 1:
            num_direct_to_keep = self.max_nodes - 1
            direct_neighbors = rng.choice(direct_neighbors, size=num_direct_to_keep, replace=False).tolist()
            sampled_other = []
        else:
            if num_to_sample > 0 and len(other_nodes) > 0:
                if len(other_nodes) > num_to_sample:
                    sampled_other = rng.choice(other_nodes, size=num_to_sample, replace=False).tolist()
                else:
                    sampled_other = other_nodes
            else:
                sampled_other = []

        return np.array([i] + direct_neighbors + sampled_other, dtype=np.int64)

    def generate_subgraph(self, G, i, item):
        cache_key_item = int(item)
        if self.subgraph_cache is not None:
            cached = self.subgraph_cache.get(cache_key_item, self.h, self.max_nodes)
            if cached is not None:
                return cached
        if item in self.subgraphs:
            return self.subgraphs[item]

        h_hops_neighbor = self._collect_neighbors(i)
        h_hops_neighbor = self._limit_neighbors(G, i, h_hops_neighbor, item)
        sub = G.subgraph(h_hops_neighbor)

        if sub.number_of_edges() == 0:
            f_hop_in = self.adj_in[int(i)]
            f_hop_out = self.adj_out[int(i)]
            direct_neighbors = list(set(f_hop_in + f_hop_out))

            if len(direct_neighbors) > 0 and len(h_hops_neighbor) < self.max_nodes:
                additional = [n for n in direct_neighbors if n not in h_hops_neighbor]
                if len(additional) > 0:
                    num_to_add = min(len(additional), self.max_nodes - len(h_hops_neighbor))
                    rng = np.random.RandomState(int(item) % 2**31)
                    if len(additional) > num_to_add:
                        additional = rng.choice(additional, size=num_to_add, replace=False).tolist()
                    h_hops_neighbor = np.concatenate([h_hops_neighbor, additional])
                    sub = G.subgraph(h_hops_neighbor)

            if sub.number_of_edges() == 0:
                print(f"Warning: subgraph of node {i} has no edges (possibly an isolated node or sampling issue)")

        h_c = list(sub.ndata[dgl.NID].numpy())
        dict_ = dict(zip(h_c, list(range(len(h_c)))))
        result = (sub, dict_[int(i)], h_c)
        self.subgraphs[item] = result
        if self.subgraph_cache is not None:
            self.subgraph_cache.put(cache_key_item, self.h, self.max_nodes, result)
        return result

    def __getitem__(self, index):
        info = [
            self.generate_subgraph(self.G[0], int(item), item)
            for sublist in self.all_batches[index]['support']
            for item in sublist
        ]
        support_x = [i for i, j, k in info]
        support_y = np.array(
            [self.subgraph2label[int(item)] for sublist in self.all_batches[index]['support'] for item in sublist]
        ).astype(np.int32)
        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        support_node_idx = [k for i, j, k in info]

        info = [
            self.generate_subgraph(self.G[0], int(item), item)
            for sublist in self.all_batches[index]['query']
            for item in sublist
        ]
        query_x = [i for i, j, k in info]
        query_y = np.array(
            [self.subgraph2label[int(item)] for sublist in self.all_batches[index]['query'] for item in sublist]
        ).astype(np.int32)
        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        support_y_relative = np.zeros(len(support_y))
        query_y_relative = np.zeros(len(query_y))
        support_y_relative[support_y == 0] = 0
        query_y_relative[query_y == 0] = 0
        support_y_relative[support_y != 0] = 1
        query_y_relative[query_y != 0] = 1

        random_idx_spt = torch.randperm(len(support_y_relative))
        support_x = [support_x[i] for i in random_idx_spt]
        support_y_relative = support_y_relative[random_idx_spt]
        support_center = support_center[random_idx_spt]
        support_node_idx = [support_node_idx[i] for i in random_idx_spt]

        random_idx_qry = torch.randperm(len(query_y_relative))
        query_x = [query_x[i] for i in random_idx_qry]
        query_y_relative = query_y_relative[random_idx_qry]
        query_center = query_center[random_idx_qry]
        query_node_idx = [query_node_idx[i] for i in random_idx_qry]

        batched_graph_spt = dgl.batch(support_x)
        batched_graph_qry = dgl.batch(query_x)
        return (
            batched_graph_spt,
            torch.LongTensor(support_y_relative),
            batched_graph_qry,
            torch.LongTensor(query_y_relative),
            torch.LongTensor(support_center),
            torch.LongTensor(query_center),
            support_node_idx,
            query_node_idx,
        )

    def __len__(self):
        return self.batchsz


def collate(samples):
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry = map(list, zip(*samples))
    return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry
