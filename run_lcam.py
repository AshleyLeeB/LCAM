from __future__ import annotations

import argparse
import csv
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import config
from dataset_configs import DatasetSpec, get_dataset_spec, resolve_dataset_keys
from train import train_single_experiment

LCAM = 'lcam'


@dataclass(frozen=True)
class VersionSpec:
    key: str
    modulation_version: str
    method_label: str
    blend_layers: str = 'all'
    blend_mode: Optional[str] = None
    modulation_aux_weight: Optional[float] = None
    alpha_bias_init: Optional[float] = None


VERSIONS: Dict[str, VersionSpec] = {
    LCAM: VersionSpec(
        key=LCAM,
        modulation_version='LCAM',
        method_label='LCAM_unfrozen',
    ),
}


def build_args_config(
    spec: VersionSpec,
    dataset: DatasetSpec,
    seed: int,
    output_dir: str,
    min_epochs: Optional[int] = None,
    early_stopping: Optional[bool] = None,
) -> Dict:
    hops = config.get('neighborhood_hops', [2])
    layers = config.get('num_gnn_layers', [2])
    cfg = {
        'data_dir': dataset.data_dir + ('/' if not dataset.data_dir.endswith('/') else ''),
        'n_way': config.get('n_way', 2),
        'k_spt': config.get('k_spt', 10),
        'k_qry': config.get('k_qry', 24),
        'input_dim': config.get('input_dim', 1),
        'hidden_dim': config.get('hidden_dim', 256),
        'h': hops[0] if isinstance(hops, list) else hops,
        'num_gnn_layers': layers[0] if isinstance(layers, list) else layers,
        'meta_lr': config.get('meta_lr', 0.003),
        'update_lr': config.get('update_lr', 0.05),
        'update_step': config.get('update_step', 10),
        'update_step_test': config.get('update_step_test', 5),
        'num_epochs': config.get('num_epochs', 10),
        'patience': config.get('patience', 4),
        'num_train_datasets': dataset.num_train_datasets,
        'max_samples_0': dataset.max_samples_0,
        'max_samples_1': dataset.max_samples_1,
        'support_other_count': config.get('support_other_count', 1),
        'query_other_count': config.get('query_other_count', 2),
        'dataloader_num_workers': config.get('dataloader_num_workers', 4),
        'dataloader_persistent_workers': config.get('dataloader_persistent_workers', True),
        'dataloader_prefetch_factor': config.get('dataloader_prefetch_factor', 4),
        'train_batchsz': config.get('train_batchsz', 50),
        'task_batch_size': config.get('task_batch_size', 4),
        'max_subgraph_nodes': config.get('max_subgraph_nodes', 100),
        'subgraph_cache_enabled': config.get('subgraph_cache_enabled', True),
        'subgraph_cache_size': config.get('subgraph_cache_size', 200000),
        'cudnn_deterministic': config.get('cudnn_deterministic', False),
        'cudnn_benchmark': config.get('cudnn_benchmark', True),
        'show_progress': config.get('show_progress', True),
        'modulation_version': spec.modulation_version,
        'freeze_gnn': False,
        'min_epochs': config.get('min_epochs', 0) if min_epochs is None else min_epochs,
        'early_stopping': config.get('early_stopping', True) if early_stopping is None else early_stopping,
        'modulation_eta': config.get('modulation_eta', 0.5),
        'modulation_gamma': config.get('modulation_gamma', 0.5),
        'modulation_pcgrad_tau': config.get('modulation_pcgrad_tau', 0.2),
        'modulation_v7_tau_d': config.get('modulation_v7_tau_d', 0.8),
        'modulation_aux_weight': (
            spec.modulation_aux_weight
            if spec.modulation_aux_weight is not None
            else config.get('modulation_aux_weight', 0.01)
        ),
        'blend_mode': (
            spec.blend_mode
            if spec.blend_mode is not None
            else config.get('blend_mode', 'original')
        ),
        'blend_layers': spec.blend_layers,
        'alpha_bias_init': (
            spec.alpha_bias_init
            if spec.alpha_bias_init is not None
            else config.get('alpha_bias_init', 0.0)
        ),
        'lambda_cml': 0.0,
        'save_gradient_analysis': False,
        'dataloader_seed': True,
        'aux_lambda': config.get('aux_lambda', 0.01),
        'gate_objective': config.get('gate_objective', 'mainpath'),
        'ablation_mode': config.get('ablation_mode', 'full'),
    }
    return cfg


def write_csv(path: str, rows: List[Dict], fieldnames: Sequence[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{digits}f}'


def compute_method_summary(
    method: str,
    rows: List[Dict],
    dataset_key: Optional[str] = None,
) -> Dict:
    ok = [
        r for r in rows
        if r['method'] == method
        and r.get('success')
        and (dataset_key is None or r.get('dataset') == dataset_key)
    ]
    auprcs = [r['AUPRC'] for r in ok]
    recalls = [r['Recall_K'] for r in ok]
    times = [r['training_time'] for r in ok if r.get('training_time') is not None]
    a_mean = float(np.mean(auprcs)) if auprcs else None
    r_mean = float(np.mean(recalls)) if recalls else None
    return {
        'method': method,
        'AUPRC_mean': a_mean,
        'AUPRC_std': float(np.std(auprcs)) if auprcs else None,
        'AUPRC_median': float(np.median(auprcs)) if auprcs else None,
        'AUPRC_min': float(np.min(auprcs)) if auprcs else None,
        'AUPRC_max': float(np.max(auprcs)) if auprcs else None,
        'Recall_mean': r_mean,
        'Recall_std': float(np.std(recalls)) if recalls else None,
        'Recall_median': float(np.median(recalls)) if recalls else None,
        'Recall_min': float(np.min(recalls)) if recalls else None,
        'Recall_max': float(np.max(recalls)) if recalls else None,
        'Time_mean': float(np.mean(times)) if times else None,
        'CV_AUPRC': float(np.std(auprcs) / a_mean) if auprcs and a_mean else None,
        'CV_Recall': float(np.std(recalls) / r_mean) if recalls and r_mean else None,
        'num_runs': len(ok),
    }


def main():
    parser = argparse.ArgumentParser(description='AMLWorld LCAM multi-seed runs')
    parser.add_argument('--start_seed', type=int, default=1)
    parser.add_argument('--end_seed', type=int, default=20)
    parser.add_argument(
        '--only',
        type=str,
        choices=[LCAM],
        default=LCAM,
        help='LCAM only',
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=None,
        help='Minimum training epochs before early stopping can trigger (default: 0)',
    )
    parser.add_argument(
        '--no_early_stop',
        action='store_true',
        help='Disable early stopping; always train for num_epochs (default 10)',
    )
    parser.add_argument(
        '--dataset',
        action='append',
        choices=['all', 'amlworld'],
        default=None,
        help='Dataset to run (default: amlworld)',
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='results',
        help='Directory for logs, CSV summaries, and per-seed run folders',
    )
    args = parser.parse_args()

    if args.start_seed > args.end_seed:
        raise ValueError(
            f'start_seed ({args.start_seed}) must be <= end_seed ({args.end_seed})'
        )
    seeds = list(range(args.start_seed, args.end_seed + 1))

    spec = VERSIONS[args.only]
    specs = [spec]

    if args.dataset is None:
        dataset_keys = ['amlworld']
    else:
        dataset_keys: List[str] = []
        for selection in args.dataset:
            for key in resolve_dataset_keys(selection):
                if key not in dataset_keys:
                    dataset_keys.append(key)
    dataset_specs = [get_dataset_spec(key) for key in dataset_keys]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_root / f'stability_20seed_log_{timestamp}.txt'
    detail_path = output_root / f'stability_20seed_detail_{timestamp}.csv'
    summary_path = output_root / f'stability_20seed_summary_{timestamp}.csv'

    model_type = config.get('model_types', ['GCN'])[0]
    detail_rows: List[Dict] = []

    def log(msg: str):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('AMLWorld LCAM evaluation\n')
        f.write(f'datasets={[d.key for d in dataset_specs]}\n')
        f.write(f'seeds={seeds}\n')
        f.write(f'version={spec.key}\n')
        if args.min_epochs is not None:
            f.write(f'min_epochs={args.min_epochs}\n')
        f.write(f'early_stopping={not args.no_early_stop}\n')

    total = len(seeds) * len(dataset_specs)
    current = 0

    log('=' * 80)
    log(f'LCAM stability — {total} runs')
    log(f'  datasets: {[d.display_name for d in dataset_specs]}')
    log(f'  seeds: {seeds}')
    log(f'  method: {spec.method_label}')
    log('=' * 80)

    for dataset in dataset_specs:
        log('\n' + '#' * 80)
        log(
            f'Dataset: {dataset.display_name} '
            f'(train={dataset.num_train_datasets}, '
            f'max_samples_0={dataset.max_samples_0}, '
            f'max_samples_1={dataset.max_samples_1})'
        )
        log('#' * 80)

        for seed in seeds:
            current += 1
            output_dir = str(
                output_root / f'stability_{dataset.key}_{spec.key}_seed_{seed}'
            )
            args_config = build_args_config(
                spec, dataset, seed, output_dir,
                min_epochs=args.min_epochs,
                early_stopping=False if args.no_early_stop else None,
            )

            log('\n' + '=' * 80)
            log(
                f'Run {current}/{total}: '
                f'{dataset.display_name}, {spec.method_label}, seed={seed}'
            )
            if args_config['min_epochs'] > 0 or not args_config['early_stopping']:
                log(
                    f"  min_epochs={args_config['min_epochs']}, "
                    f"early_stopping={args_config['early_stopping']}"
                )
            try:
                result = train_single_experiment(seed, model_type, args_config)
                row = {
                    'dataset': dataset.key,
                    'dataset_name': dataset.display_name,
                    'method': spec.method_label,
                    'version_key': spec.key,
                    'seed': seed,
                    'min_epochs': args_config.get('min_epochs'),
                    'AUPRC': result.get('auprc'),
                    'Recall_K': result.get('recall_at_k'),
                    'training_time': result.get('elapsed_time'),
                    'success': result.get('success'),
                }
                detail_rows.append(row)
                if result.get('success'):
                    log(
                        f"  AUPRC={result['auprc']:.4f}, "
                        f"Recall@K={result['recall_at_k']:.4f}, "
                        f"time={result.get('time_str', result.get('elapsed_time'))}"
                    )
                else:
                    log('  FAILED')
            except Exception as exc:
                log(f'  ERROR: {exc}')
                with open(log_path, 'a', encoding='utf-8') as f:
                    traceback.print_exc(file=f)
                detail_rows.append({
                    'dataset': dataset.key,
                    'dataset_name': dataset.display_name,
                    'method': spec.method_label,
                    'version_key': spec.key,
                    'seed': seed,
                    'min_epochs': args_config.get('min_epochs'),
                    'AUPRC': None,
                    'Recall_K': None,
                    'training_time': None,
                    'success': False,
                })

    write_csv(
        detail_path,
        [
            {
                'dataset': r.get('dataset', ''),
                'dataset_name': r.get('dataset_name', ''),
                'method': r['method'],
                'version_key': r.get('version_key', ''),
                'seed': r['seed'],
                'min_epochs': r.get('min_epochs', ''),
                'AUPRC': fmt_float(r['AUPRC']),
                'Recall_K': fmt_float(r['Recall_K']),
                'training_time': fmt_float(r['training_time'], 1),
                'success': r.get('success'),
            }
            for r in detail_rows
        ],
        [
            'dataset', 'dataset_name', 'method', 'version_key', 'seed', 'min_epochs',
            'AUPRC', 'Recall_K', 'training_time', 'success',
        ],
    )

    summary_rows = []
    for dataset in dataset_specs:
        s = compute_method_summary(spec.method_label, detail_rows, dataset.key)
        summary_rows.append({
            'dataset': dataset.key,
            'dataset_name': dataset.display_name,
            'method': s['method'],
            'AUPRC_mean': fmt_float(s['AUPRC_mean']),
            'AUPRC_std': fmt_float(s['AUPRC_std']),
            'AUPRC_median': fmt_float(s['AUPRC_median']),
            'AUPRC_min': fmt_float(s['AUPRC_min']),
            'AUPRC_max': fmt_float(s['AUPRC_max']),
            'Recall_mean': fmt_float(s['Recall_mean']),
            'Recall_std': fmt_float(s['Recall_std']),
            'Recall_median': fmt_float(s['Recall_median']),
            'Recall_min': fmt_float(s['Recall_min']),
            'Recall_max': fmt_float(s['Recall_max']),
            'Time_mean': fmt_float(s['Time_mean'], 1),
            'CV_AUPRC': fmt_float(s['CV_AUPRC']),
            'CV_Recall': fmt_float(s['CV_Recall']),
        })
        print(
            f"\n{dataset.display_name} | {spec.method_label}: "
            f"AUPRC={fmt_float(s['AUPRC_mean'])}±{fmt_float(s['AUPRC_std'])}, "
            f"Recall={fmt_float(s['Recall_mean'])}±{fmt_float(s['Recall_std'])}"
        )

    write_csv(
        summary_path,
        summary_rows,
        [
            'dataset', 'dataset_name', 'method',
            'AUPRC_mean', 'AUPRC_std', 'AUPRC_median', 'AUPRC_min', 'AUPRC_max',
            'Recall_mean', 'Recall_std', 'Recall_median', 'Recall_min', 'Recall_max',
            'Time_mean', 'CV_AUPRC', 'CV_Recall',
        ],
    )

    log(f'\nDetail: {detail_path}')
    log(f'Summary: {summary_path}')
    log(f'Log: {log_path}')


if __name__ == '__main__':
    main()
