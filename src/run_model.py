import wandb
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import src.gcn_model as gcn_model
from src.protein_graph import pncaGraph
import src.model_helpers as model_helpers
from typing import Union, List

def _run_pnca_gcn_training(
    full_dataset: List[Data],
    num_node_features: int,
    hidden_channels: int,
    output_channels: int,
    dropout: float,
    batch_size: int,
    train_split: float,
    test_split: float,
    learning_rate: float,
    wd: float,
    epochs: int,
    lr_scheduling: bool,
    early_stop: bool,
    save_path: str,
    wandb_params: dict
):
    """
    Helper to set up dataloaders, model, optimizer, scheduler, trainer, and run training.
    """
    
    if torch.cuda.is_available():
        print('Using CUDA')
        model_device = 'cuda'  
    else:
        print('Using CPU')
        model_device = 'cpu'

    # Create DataLoaders
    train_loader, test_loader, val_loader, dataset_dict = gcn_model.load(
        dataset=full_dataset,
        batch_size=batch_size,
        shuffle_dataset=False,
        train_split=train_split,
        test_split=test_split,
        val_split=0
    )

    # Set up GCN model
    model = gcn_model.GCN(
        input_channels=num_node_features,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        p=dropout
    ).to(model_device)

    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict = dataset_dict

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.5, patience=10, verbose=True
    ) if lr_scheduling else None

    criterion = torch.nn.BCEWithLogitsLoss() if output_channels == 1 else torch.nn.CrossEntropyLoss()

    gcntrainer = gcn_model.GCNTrainer(
        model=model,
        loss_func=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        output_dim=output_channels
    )

    if wandb_params['use_wandb']:
        wandb.init(
            project=wandb_params['wandb_project'],
            name=wandb_params['wandb_name'],
            config=wandb_params.get('config', {})
        )

    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(
        epochs=epochs,
        use_wandb=wandb_params.get('use_wandb', False) or wandb_params.get('sweep', False),
        path=save_path,
        early_stop={'patience': 20, 'min_delta': 0} if early_stop else False
    )

    return model

def pnca_simpleGCN(
    self_loops: bool,
    cutoff_distance: float,
    edge_weight_func: str,
    batch_size: int,
    num_node_features: int,
    hidden_channels: int,
    learning_rate: float,
    wd: float,
    epochs: int,
    sequences: Union[pd.DataFrame, dict] = None,
    dataset: List[Data] = None,
    output_channels: int = 2,
    normalise_ews: bool = True,
    lambda_param: float = None,
    dropout = 0.5,
    lr_scheduling = False,
    early_stop = True,
    save_path: str = None,
    wandb_params: dict = {'use_wandb': False, 'wandb_project': None, 'wandb_name': None, 'sweep': False}
    ):
    """
    Runs PncA GCN model pipeline. Sequence datasets must be generated prior.
    
    Args:
        self_loops (bool): Include self loops in graph.
        cutoff_distance (float): Distance cutoff in Angstroms for edges in the graph.
        edge_weight_func (str): Edge weight function.
        batch_size (int): Batch size for training.
        num_node_features (int): Number of node features / size of input channel.
        hidden_channels (int): Number of hidden channels.
        learning_rate (float): Learning rate.
        wd (float): Weight decay.
        epochs (int): Number of epochs in training.
        sequences (Union[pd.DataFrame, dict], optional): If train/test split already done, provide dict in form of {'train': train_df, 'test': test_df}. Can be None if dataset is provided. Defaults to None.
        dataset (List[torch_geometric.data.Data], optional): Full dataset. If provided, sequences will be ignored. Defaults to None.
        output_channels (int, optional): Number of output channels. Defaults to 2.
        normalise_ews (bool, optional): Whether to normalise edge weights. Defaults to True.
        lambda_param (float, optional): Lambda parameter for exponential edge weighting. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        lr_scheduling (bool, optional): Use learning rate scheduling. Defaults to False.
        early_stop (bool, optional): Use early stopping in model training. Defaults to True.
        save_path (str, optional): Path to save model. Defaults to None.
        wandb_params (dict, optional): Weights & Biases logging parameters. Should include keys 'use_wandb', 'wandb_project', 'wandb_name', and 'sweep'. Defaults to {'use_wandb': False, 'wandb_project': None, 'wandb_name': None, 'sweep': False}.

    Returns:
        model (torch.nn.Module): Trained GCN model
    """
    # Create graph
    pnca = pncaGraph(pdb='../pdb/3PL1-PZA.pdb',
                    lig_resname='PZA', 
                    self_loops=self_loops,
                    cutoff_distance=cutoff_distance)
    
    if dataset is None:
        wt_seq = 'MRALIIVDVQNDFCEGGSLAVTGGAALARAISDYLAEAADYHHVVATKDFHIDPGDHFSGTPDYSSSWPPHCVSGTPGADFHPSLDTSAIEAVFYKGAYTGAYSGFEGVDENGTPLLNWLRQRGVDEVDVVGIATDHCVRQTAEDAVRNGLATRVLVDLTAGVSADTTVAALEEMRTASVELVCS'
        if type(sequences) == pd.DataFrame:
            # generate pyg Dataset
            pnca.gen_dataset(wt_seq=wt_seq,
                        sequences=sequences,
                        edge_weights=edge_weight_func,
                        lambda_param=lambda_param,
                        normalise=normalise_ews,             
                        )
            train_split, test_split = 0.7, 0.3
            full_dataset = pnca.dataset
        elif type(sequences) == dict:
            assert 'test' and 'train' in sequences.keys(), "Please provide a dictionary with keys 'train' and 'test'"
            pnca.gen_dataset(wt_seq=wt_seq,
                        sequences=sequences['train'],
                        edge_weights=edge_weight_func,
                        lambda_param=lambda_param,
                        normalise=normalise_ews,             
                        )
            train_dataset = pnca.dataset
            pnca.gen_dataset(wt_seq=wt_seq,
                        sequences=sequences['test'],
                        edge_weights=edge_weight_func,
                        lambda_param=lambda_param,
                        normalise=normalise_ews,             
                        )
            test_dataset = pnca.dataset
            full_dataset = train_dataset + test_dataset
            train_split = len(train_dataset) / len(full_dataset)
            test_split = len(test_dataset) / len(full_dataset)
        else:
            raise ValueError("Please provide a pandas DataFrame or a dictionary with keys 'train' and 'test'")
    else:
        full_dataset = dataset
        train_split, test_split = 0.7, 0.3
    
    if wandb_params.get('use_wandb', False) or wandb_params.get('sweep', False):    
        wandb_params['config'] = {
                "num_node_features": num_node_features,
                "hidden_channels": hidden_channels,
                "learning_rate": learning_rate,
                "weight_decay": wd,
                "cutoff_distance": cutoff_distance,
                "self_loops": self_loops,
                "lambda_param": lambda_param,
                "dropout": dropout,
                "n_samples": wandb_params['n_samples'],
                "batch_size": batch_size,
                "epochs": epochs,
            }

    return _run_pnca_gcn_training(
        full_dataset=full_dataset,
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        dropout=dropout,
        batch_size=batch_size,
        train_split=train_split,
        test_split=test_split,
        learning_rate=learning_rate,
        wd=wd,
        epochs=epochs,
        lr_scheduling=lr_scheduling,
        early_stop=early_stop,
        save_path=save_path,
        wandb_params=wandb_params
    )

def pnca_GCN_vary_graph(
    self_loops: bool,
    cutoff_distance: float,
    edge_weight_func: str,
    batch_size: int,
    num_node_features: int,
    hidden_channels: int,
    learning_rate: float,
    wd: float,
    epochs: int,
    graph_dict: dict ,
    output_channels: int = 2,
    normalise_ews: bool = True,
    lambda_param: float = None,
    dropout: float = 0.5,
    lr_scheduling: bool = False,
    early_stop: bool = True,
    recreate_graph: bool = False,
    shuffle_edges: bool = False,
    no_node_mpfs: bool = False,
    no_node_chem_feats: bool = False,
    rand_node_feats: bool = False,
    save_path: str = None,
    wandb_params: dict = {'use_wandb': False, 'wandb_project': None, 'wandb_name': None, 'sweep': False}
    ):
    """
    Runs PncA GCN model pipeline. 
    Input in the form of nested dictionary with keys 'train' and 'test', then a key for each sample with a pncaGraph object.
    
    Args:
        self_loops (bool): Include self loops in graph.
        cutoff_distance (float): Distance cutoff in Angstroms for edges in the graph.
        edge_weight_func (str): Edge weight function.
        batch_size (int): Batch size for training.
        num_node_features (int): Number of node features / size of input channel.
        hidden_channels (int): Number of hidden channels.
        learning_rate (float): Learning rate.
        wd (float): Weight decay.
        epochs (int): Number of epochs in training.
        graph_dict (dict): Provide dict in form of:
            {
                'train': {
                    'graph': <src.protein_graph.pncaGraph object>,
                    'metadata': ...,
                    ...
                }, 
                'test': {
                    'graph': <src.protein_graph.pncaGraph object>,
                    'metadata': ...,
                    ...
                }
            }
        output_channels (int, optional): Number of output channels. Defaults to 2.
        normalise_ews (bool, optional): Normalise edge weights. Defaults to True.
        lambda_param (float, optional): Lambda parameter for exponential edge weighting.
            Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        lr_scheduling (bool, optional): Use learning rate scheduling. Defaults to False.
        early_stop (bool, optional): Use early stopping in model training. Defaults to True.
        recreate_graph (bool, optional): Whether to redefine edge index and edge weights in
            the graph. To be used if desired graph structure (e.g. cutoff distance) is
            different from the passed in graph_dict, or if performing a sweep where cutoff
            distance is varied. Defaults to False.
        shuffle_edges (bool, optional): Whether to shuffle edges in the graph. Use as part
            of control test to evaluate effect of graph structure. Defaults to False.
        no_node_mpfs (bool, optional): Whether to remove meta-predictor features for
            control test. Defaults to False.
        no_node_chem_feats (bool, optional): Whether to remove chemical features (from
            SBMLCore) for control test. Defaults to False.
        rand_node_feats (bool, optional): Whether to use random node features for control
            test. Defaults to False.
        save_path (str, optional): Path to save the trained model. Defaults to None.
        wandb_params (dict, optional): Dictionary to provide parameters for WandB run.
            Defaults to {'use_wandb': False, 'wandb_project': None, 'wandb_name': None,
            'sweep': False}.
        
    Returns:
        model (torch.nn.Module): Trained GCN model
    """
    
    if any([shuffle_edges, no_node_mpfs, no_node_chem_feats, rand_node_feats, recreate_graph]):
        # redefine graph_dict
        model_helpers.redefine_graph(graph_dict,
                        cutoff_distance=cutoff_distance,
                        edge_weight_func=edge_weight_func,
                        normalise_ews=normalise_ews,
                        lambda_param=lambda_param,
                        recreate_graph=recreate_graph,
                        no_node_mpfs=no_node_mpfs,
                        no_node_chem_feats=no_node_chem_feats,
                        rand_node_feats=rand_node_feats,
                        shuffle_edges=shuffle_edges)

    train_split, test_split = 0.7, 0.3
    # create dataset list
    full_dataset = []
    for sample in graph_dict['train']:
        full_dataset.append(graph_dict['train'][sample]['graph'].dataset[0])
    for sample in graph_dict['test']:
        full_dataset.append(graph_dict['test'][sample]['graph'].dataset[0])

    if wandb_params.get('use_wandb', False) or wandb_params.get('sweep', False):
        wandb_params['config'] = {
                "num_node_features": num_node_features,
                "hidden_channels": hidden_channels,
                "learning_rate": learning_rate,
                "weight_decay": wd,
                "cutoff_distance": cutoff_distance,
                "self_loops": self_loops,
                "lambda_param": lambda_param,
                "dropout": dropout,
                "n_samples": wandb_params['n_samples'],
                "batch_size": batch_size,
                "epochs": epochs,
            }
        
    return _run_pnca_gcn_training(
        full_dataset=full_dataset,
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        dropout=dropout,
        batch_size=batch_size,
        train_split=train_split,
        test_split=test_split,
        learning_rate=learning_rate,
        wd=wd,
        epochs=epochs,
        lr_scheduling=lr_scheduling,
        early_stop=early_stop,
        save_path=save_path,
        wandb_params=wandb_params
    ) 