config = {
    'data_dir': './Data/AMLWorld/Data/',

    'n_way': 2,
    'k_spt': 10,
    'k_qry': 24,
    'input_dim': 1,
    'hidden_dim': 256,

    'neighborhood_hops': [2],
    'num_gnn_layers': [2],

    'meta_lr': 0.003,
    'update_lr': 0.05,
    'update_step': 10,
    'update_step_test': 5,

    'num_seeds': 3,
    'num_epochs': 10,
    'patience': 4,
    'num_train_datasets': 6,

    'max_samples_0': 9800,
    'max_samples_1': 150,
    'support_other_count': 1,
    'query_other_count': 2,

    'model_types': ['GCN'],
    'seeds': [41, 43, 44, 45],
    'freeze_gnn': True,

    'modulation_version': 'LCAM',
    'blend_mode': 'original',
    'blend_layers': 'all',
    'modulation_aux_weight': 0.01,
    # Same as LCAM/DTML LCAM-1.py --only mainpath (V3_blend + gate_objective=mainpath)
    'aux_lambda': 0.01,
    'gate_objective': 'mainpath',
    'alpha_bias_init': 0.0,
    'ablation_mode': 'full',

    'train_batchsz': 50,
    'task_batch_size': 4,
    'max_subgraph_nodes': 100,
    'subgraph_cache_enabled': True,
    'subgraph_cache_size': 200000,
    'cudnn_deterministic': False,
    'cudnn_benchmark': True,
    'show_progress': True,

    'dataloader_num_workers': 4,
    'dataloader_persistent_workers': True,
    'dataloader_prefetch_factor': 4,
}
