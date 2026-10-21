import torch, os
from subgraph_data_processing import Subgraphs, FlattenedTaskDataset, meta_collate_fn
from subgraph_cache import SubgraphCache
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import networkx as nx
import numpy as np
from tqdm import tqdm
import dgl
from meta import Meta
import time
import copy
import warnings
import random, sys, pickle
import traceback
from itertools import zip_longest
from config import config
warnings.filterwarnings("ignore", category=UserWarning, module="dgl")
warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def collate(samples): 
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry = map(list, zip(*samples))
        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry

def normalize_model_type(model_type: str) -> str:
    if model_type.startswith('GCN'):
        return 'GCN'
    return model_type


SCENARIO = 'GNN++'


def train_single_experiment(seed, model_type, args_config):
    start_time = time.time()
    
    torch.manual_seed(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    dgl.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn_deterministic = args_config.get('cudnn_deterministic', False)
    cudnn_benchmark = args_config.get('cudnn_benchmark', True)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark and torch.cuda.is_available()
    
    root = args_config['data_dir']
    
    feat = np.load(root + 'features_5.npy', allow_pickle = True)
    with open(root + 'new_graph_guiyihua.pkl', 'rb') as f:
        dgl_graph = pickle.load(f)  
    with open(root + 'label.pkl', 'rb') as f:
        label = pickle.load(f)
    if len(feat.shape) == 2:
        feat = [feat]
    
    actual_model_type = normalize_model_type(model_type)

    h = args_config['h']
    num_gnn_layers = args_config.get('num_gnn_layers', 2)
    hidden_dim = args_config['hidden_dim']
    labels_num = 2
    model_config = [('GraphConv', [feat[0].shape[1], hidden_dim])]
    if num_gnn_layers > 1:
        model_config += [('GraphConv', [hidden_dim, hidden_dim])] * (num_gnn_layers - 1)
    model_config += [('Linear', [hidden_dim, labels_num])]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    args = Args(args_config)
    maml = Meta(args, model_config).to(device) 

    num_train_datasets = args_config.get('num_train_datasets', 2)
    max_samples_0 = args_config.get('max_samples_0', 4500)
    max_samples_1 = args_config.get('max_samples_1', 200)
    support_other_count = args_config.get('support_other_count', 1)
    query_other_count = args_config.get('query_other_count', 2)
    train_batchsz = args_config.get('train_batchsz', 50)
    task_batch_size = args_config.get('task_batch_size', 4)
    max_subgraph_nodes = args_config.get('max_subgraph_nodes', 100)
    subgraph_cache = SubgraphCache(max_size=args_config.get('subgraph_cache_size', 200000)) if args_config.get('subgraph_cache_enabled', True) else None

    common_dataset_kwargs = dict(
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        adjs=dgl_graph,
        h=h,
        task_labels=['0', '1'],
        max_nodes=max_subgraph_nodes,
        max_samples_0=max_samples_0,
        max_samples_1=max_samples_1,
        subgraph_cache=subgraph_cache,
    )

    train_datasets = []
    for i in range(1, num_train_datasets + 1):
        train_dataset = Subgraphs(
            root,
            f'train_{i}',
            label,
            batchsz=train_batchsz,
            support_other_count=support_other_count,
            query_other_count=query_other_count,
            **common_dataset_kwargs,
        )
        train_datasets.append(train_dataset)

    db_task_5 = Subgraphs(root, 'val', label, batchsz=1, **common_dataset_kwargs)
    db_task_6 = Subgraphs(root, 'test', label, batchsz=1, **common_dataset_kwargs)
    flat_train_dataset = FlattenedTaskDataset(train_datasets)

    max_recall = 0
    best_model_state = None
    
    patience = args_config.get('patience', 4)
    min_epochs = args_config.get('min_epochs', 0)
    early_stopping = args_config.get('early_stopping', True)
    no_improve_count = 0
    best_val_recall = 0.0
    train_loss_history = []

    db_task_5.extract_batch(db_task_5.batchsz, db_task_5.dictLabels)
    db_task_6.extract_batch(db_task_6.batchsz, db_task_6.dictLabels)

    num_workers = args_config.get('dataloader_num_workers', 4)
    persistent_workers = args_config.get('dataloader_persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = args_config.get('dataloader_prefetch_factor', 2) if num_workers > 0 else 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        flat_train_dataset,
        batch_size=task_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=meta_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=torch.Generator().manual_seed(seed) if args_config.get('dataloader_seed', False) else None,
    )
    db_5 = DataLoader(
        db_task_5,
        1,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    db_6 = DataLoader(
        db_task_6,
        1,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    num_epochs = args_config.get('num_epochs', 10)
    for k in range(num_epochs):
        epoch_start = time.time()
        maml.reset_blend_epoch_stats()

        meta_step = 0
        for train_dataset in train_datasets:
            train_dataset.extract_batch(train_dataset.batchsz, train_dataset.dictLabels)
        flat_train_dataset.rebuild_task_map()

        auprc_all_train = []
        recall_at_k_all_train = []
        losses_all_train = []

        train_iter = tqdm(
            train_loader,
            desc=f'Epoch {k + 1}/{num_epochs}',
            leave=False,
        ) if args_config.get('show_progress', True) else train_loader

        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry in train_iter:
            try:
                maml.set_training_context(epoch=k, meta_step=meta_step)
                accs, loss, auprc, recall_at_k = maml(
                    x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat
                )
                meta_step += 1
                auprc_all_train.append(auprc[-1] if isinstance(auprc, (list, np.ndarray)) else auprc)
                recall_at_k_all_train.append(recall_at_k[-1] if isinstance(recall_at_k, (list, np.ndarray)) else recall_at_k)
                if loss is not None:
                    losses_all_train.append(loss.item() if hasattr(loss, 'item') else loss)
            except Exception as e:
                print(f"训练批次出错: {e}")
                traceback.print_exc()
                raise

        if len(losses_all_train) > 0:
            train_loss_history.append(np.mean(losses_all_train))
        epoch_msg = (
            f'Epoch {k + 1}/{num_epochs} train done in {time.time() - epoch_start:.1f}s | '
            f'loss={train_loss_history[-1]:.4f}' if losses_all_train else
            f'Epoch {k + 1}/{num_epochs} train done in {time.time() - epoch_start:.1f}s'
        )
        print(epoch_msg)
        blend_log = maml.format_blend_epoch_stats()
        if blend_log:
            print(blend_log)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        auprc_all_val = []
        recall_at_k_all_val = []
        
        val_count = 0
        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry in db_5:
            val_count += 1
            try:
                if val_count % 10 == 1 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                accs, auprc, recall_at_k = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat)
                auprc_all_val.append(auprc[-1] if isinstance(auprc, (list, np.ndarray)) else auprc)
                recall_at_k_all_val.append(recall_at_k[-1] if isinstance(recall_at_k, (list, np.ndarray)) else recall_at_k)
            except Exception as e:
                print(f"验证批次 {val_count} 出错: {e}")
                traceback.print_exc()
                raise

        auprc_val = np.mean(auprc_all_val) if len(auprc_all_val) > 0 else 0.0
        recall_at_k_val = np.mean(recall_at_k_all_val) if len(recall_at_k_all_val) > 0 else 0.0

        if recall_at_k_val > best_val_recall:
            improvement = recall_at_k_val - best_val_recall
            best_val_recall = recall_at_k_val
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if recall_at_k_val > max_recall:
            max_recall = recall_at_k_val
            best_model_state = {k: v.detach().cpu().clone() for k, v in maml.state_dict().items()}
        
        if early_stopping and no_improve_count >= patience and (k + 1) >= min_epochs:
            break
    
    if best_model_state is not None:
        maml.load_state_dict(best_model_state)

    if subgraph_cache is not None:
        cache_stats = subgraph_cache.stats()
        print(
            f'Subgraph cache: size={cache_stats["size"]}, '
            f'hit_rate={cache_stats["hit_rate"]:.1f}%'
        )

    auprc_all_test_max = []
    recall_at_k_all_test_max = []
    test_count = 0
    test_sample_count = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    test_start_time = time.time()
    
    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry in db_6:
        test_count += 1
        try:
            if isinstance(y_qry, list):
                batch_sample_count = sum(len(y) for y in y_qry)
            elif isinstance(y_qry, torch.Tensor):
                batch_sample_count = y_qry.shape[0] if len(y_qry.shape) > 0 else 1
            else:
                batch_sample_count = len(y_qry) if hasattr(y_qry, '__len__') else 1
            test_sample_count += batch_sample_count
            
            if test_count % 10 == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            accs, auprc, recall_at_k = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, feat)
            auprc_all_test_max.append(auprc[-1] if isinstance(auprc, (list, np.ndarray)) else auprc)
            recall_at_k_all_test_max.append(recall_at_k[-1] if isinstance(recall_at_k, (list, np.ndarray)) else recall_at_k)
        except Exception as e:
            print(f"测试批次 {test_count} 出错: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    test_elapsed_time = time.time() - test_start_time
    test_hours = int(test_elapsed_time // 3600)
    test_minutes = int((test_elapsed_time % 3600) // 60)
    test_seconds = int(test_elapsed_time % 60)
    test_time_str = f'{test_hours:02d}:{test_minutes:02d}:{test_seconds:02d}'
    
    if test_sample_count > 0:
        test_time_per_sample_ms = (test_elapsed_time * 1000) / test_sample_count
    else:
        test_time_per_sample_ms = 0.0
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    if len(auprc_all_test_max) > 0:
        auprc_test_max = np.mean(auprc_all_test_max)
        recall_at_k_test_max = np.mean(recall_at_k_all_test_max)
        return {
            'auprc': auprc_test_max,
            'recall_at_k': recall_at_k_test_max,
            'success': True,
            'elapsed_time': elapsed_time,
            'time_str': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
            'test_elapsed_time': test_elapsed_time,
            'test_time_str': test_time_str,
            'test_sample_count': test_sample_count,
            'test_time_per_sample_ms': test_time_per_sample_ms,
            'scenario': SCENARIO,
            'model_type': model_type,
            'actual_model_type': actual_model_type,
            'model_state': best_model_state,
            'modulation_version': args_config.get('modulation_version', 'LCAM'),
            'modulation_gamma': args_config.get('modulation_gamma', 0.5),
            'freeze_gnn': args_config.get('freeze_gnn', True),
            'skip_reason': None
        }
    else:
        return {
            'auprc': None,
            'recall_at_k': None,
            'success': False,
            'elapsed_time': elapsed_time,
            'time_str': f'{hours:02d}:{minutes:02d}:{seconds:02d}',
            'test_elapsed_time': test_elapsed_time if 'test_elapsed_time' in locals() else None,
            'test_time_str': test_time_str if 'test_time_str' in locals() else None,
            'test_sample_count': test_sample_count if 'test_sample_count' in locals() else 0,
            'test_time_per_sample_ms': test_time_per_sample_ms if 'test_time_per_sample_ms' in locals() else 0.0,
            'scenario': SCENARIO,
            'model_type': model_type,
            'actual_model_type': actual_model_type,
            'freeze_gnn': args_config.get('freeze_gnn', True),
            'skip_reason': None
        }

def main():
    from run_lcam import main as stability_main
    stability_main()


if __name__ == '__main__':
    main()
