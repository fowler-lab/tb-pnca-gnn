import random
import numpy as np
import pickle as pkl
import torch
import warnings
import yaml
import argparse
import wandb
from omegaconf import OmegaConf
from src import run_model


def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def train(config, graph_dict):
    """Run a single training run with the given config."""
    n_samples = len(graph_dict['train']) + len(graph_dict['test'])

    # Params
    cutoff_distance = config['cutoff_distance']
    num_node_features = config['num_node_features']
    batch_size = config['batch_size']
    hidden_channels = config['hidden_channels']
    dropout = config['dropout']
    edge_weight_func = config['edge_weight_func']
    edge_weight_lambda = config['edge_weight_lambda']
    learning_rate = config['learning_rate']
    wd = config['weight_decay']
    epochs = config['epochs']
    project = config['project']
    run_name = config['run_name']

    # Train model
    model = run_model.pnca_GCN_vary_graph(
        self_loops = config.get('self_loops', False),
        cutoff_distance = cutoff_distance,
        edge_weight_func = edge_weight_func,
        batch_size = batch_size,
        num_node_features = num_node_features,
        hidden_channels = hidden_channels,
        learning_rate = learning_rate,
        wd = wd,
        dropout = dropout,
        lr_scheduling = config.get('lr_scheduling', False),
        epochs = epochs,
        graph_dict = graph_dict,
        normalise_ews = config.get('normalise_ews', True),
        lambda_param = edge_weight_lambda,
        early_stop = config.get('early_stop', False),
        recreate_graph = config.get('recreate_graph', False),
        save_path = config.get('save_path', None),
        wandb_params = {
            'use_wandb': config.get('use_wandb', True),
            'wandb_project': project,
            'wandb_name': run_name,
            'n_samples': n_samples,
            'sweep': config.get('sweep', False)
        }
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GCN model with YAML config.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--sweep_config', type=str, default=None, help='Path to sweep config YAML file (required if sweep: true in config)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset
    dataset_path = config['dataset_path']
    with open(dataset_path, 'rb') as f:
        graph_dict = pkl.load(f)

    if config.get('sweep', False):
        # Sweep mode
        if args.sweep_config is None:
            raise ValueError("--sweep_config is required when sweep: true in config.yaml")
        sweep_config = load_config(args.sweep_config)
        sweep_id = wandb.sweep(sweep=sweep_config, project=config['project'])
        print(f"Initialized sweep with ID: {sweep_id}")

        def sweep_train():
            with wandb.init() as run:
                # Merge base config with wandb sweep config (sweep overrides base)
                # wandb.config contains the parameters selected by the sweep agent
                run_config = {**config, **wandb.config}
                
                # Ensure the local config variable used for training uses the merged parameters
                train(run_config, graph_dict)

        print("Starting wandb agent...")
        wandb.agent(sweep_id, function=sweep_train)
    else:
        # Normal training
        train(config, graph_dict)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('CUDA available:', torch.cuda.is_available())
    main()
