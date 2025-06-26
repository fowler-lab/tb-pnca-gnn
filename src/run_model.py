import sbmlsim
from src.protein_graph import pncaGraph
import wandb
import src.gcn_model as gcn_model
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

from typing import Union, List

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
    dropout = 0.5,
    lr_scheduling = False,
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
        wd (float): Weight decay
        epochs (int): Number of epochs in training.
        sequences (Union[pd.DataFrame, dict]): If train/test split already done, provide dict in form of {'train': train_df, 'test': test_df}. Can be None if dataset is provided.
        dataset (torch_geometric.data.Data): Full dataset. If provided, sequences will be ignored.
        output_channels (int, optional): Number of output channels. Defaults to 2.
        normalise_ews (bool, optional): _description_. Defaults to True.
        wandb_params (dict, optional): Dictionary to provide parameters for WandB run. Defaults to use_wandb = False.

    Returns:
        model (torch.nn.Module): Trained GCN model
        train_acc (list): List of training accuracies
        test_acc (list): List of test accuracies
        train_loss (list): List of training losses
        test_loss (list): List of test losses
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
                        normalise=normalise_ews,             
                        )
            
            train_split, test_split = 0.8, 0.2
            
        elif type(sequences) == dict:
            assert 'test' and 'train' in sequences.keys(), "Please provide a dictionary with keys 'train' and 'test'"
            
            pnca.gen_dataset(wt_seq=wt_seq,
                        sequences=sequences['train'],
                        edge_weights=edge_weight_func,
                        normalise=normalise_ews,             
                        )
            
            train_dataset = pnca.dataset
            
            pnca.gen_dataset(wt_seq=wt_seq,
                        sequences=sequences['test'],
                        edge_weights=edge_weight_func,
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
        
        if edge_weight_func != 'none':
        # attach edge weights and correct edge index for varying cutoff distance

            print(f'Adjusting edge index and attaching edge weights for cutoff distance {cutoff_distance}')
            for sample in full_dataset:
                sample.edge_index = pnca.edge_index
                sample.edge_attr = pnca.calc_edge_weights(edge_weight_func, pnca.edge_dists)
                if normalise_ews:
                    sample.edge_attr = pnca.process_edge_weights(sample.edge_attr)
                    
        train_split, test_split = 0.7, 0.3
        
        
    # Create DataLoaders for train and test set
    train_loader,test_loader,val_loader, dataset_dict = gcn_model.load(dataset=full_dataset,
                                        batch_size=batch_size,
                                        shuffle_dataset=False,
                                        train_split = train_split,
                                        test_split= test_split,
                                        val_split = 0)
        
    # Set up GCN model
    model = gcn_model.GCN(
        input_channels= num_node_features,
        hidden_channels= hidden_channels,
        output_channels= output_channels,
        p=dropout
        )
    
    if torch.cuda.is_available():
        # print('Using CUDA (all available GPUs)')
        print('Using CUDA')
        
        # model = DistributedDataParallel(model)
        model = model.to('cuda')
        
    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict= dataset_dict

    # Define optimizer and loss function
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=wd
        )
    
    #* Try lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.5,
        patience=10,
        verbose=True
        ) if lr_scheduling else None
    
    if output_channels == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Set up trainer
    gcntrainer = gcn_model.GCNTrainer(model=model,
                                loss_func=criterion,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                scheduler=scheduler,
                                output_dim=output_channels
                                )
    
    if wandb_params['use_wandb']:

        wandb.init(
            # set the wandb project where this run will be logged
            project = wandb_params['wandb_project'],
            name = wandb_params['wandb_name'],
            
            # track hyperparameters and run metadata
            config={
            "num_node_features": num_node_features,
            "hidden_channels": hidden_channels,
            "learning_rate": learning_rate,
            "weight_decay": wd,
            "cutoff_distance": cutoff_distance,
            "self_loops": self_loops,
            # "n_res": 1,
            # "n_sus": 1,
            "n_samples": wandb_params['n_samples'],
            "batch_size": batch_size,
            "epochs": epochs,
            }
        )
    
    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(epochs=epochs,
                                                            use_wandb=wandb_params['use_wandb'] or wandb_params['sweep'],
                                                            # early_stop=False
                                                            early_stop={
                                                                'patience': 20, 
                                                                'min_delta': 0
                                                                }
                                                            )
    
    # return model, train_acc, test_acc, train_loss, test_loss
    return model

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
    dropout = 0.5,
    lr_scheduling = False,
    early_stop = True,
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
        wd (float): Weight decay
        epochs (int): Number of epochs in training.
        graph_dict (dict): Provide dict in form of {
            'train': {
                'graph': <src.protein_graph.pncaGraph object, 'metadata': ...,
                'graph': <src.protein_graph.pncaGraph object, 'metadata': ...
                }, 
            'test': {
                'graph': <src.protein_graph.pncaGraph object, 'metadata': ...,
                'graph': <src.protein_graph.pncaGraph object, 'metadata': ...
                }
                }    
        output_channels (int, optional): Number of output channels. Defaults to 2.
        normalise_ews (bool, optional): Normalise edge w. Defaults to True.
        wandb_params (dict, optional): Dictionary to provide parameters for WandB run. Defaults to use_wandb = False.

    Returns:
        model (torch.nn.Module): Trained GCN model
    """

    
    if edge_weight_func != 'none':
    # attach edge weights and correct edge index for varying cutoff distance

        print(f'Adjusting edge index and attaching edge weights for cutoff distance {cutoff_distance}')
        
        for sample_set in graph_dict:
            for sample in graph_dict[sample_set]:
                graph = graph_dict[sample_set][sample]['graph']
                
                # reset graph attributes based on new cutoff distance
                graph.cutoff_distance = cutoff_distance
                
                # calc new edge index and edge weights
                edge_index, d_array = graph._get_protein_struct_edges(graph.nodes.center_of_mass(compound='residues'))
                edge_dists = graph._gen_edge_dists(edge_index, d_array)
                edge_attr = graph.calc_edge_weights(edge_weight_func, edge_dists, lambda_param)
                
                if normalise_ews:
                    edge_attr = graph.process_edge_weights(edge_attr)
                
                # change edge index and edge weights for Data object
                graph_dict[sample_set][sample]['graph'].dataset[0].edge_index = edge_index
                graph_dict[sample_set][sample]['graph'].dataset[0].edge_attr = edge_attr
        
    train_split, test_split = 0.7, 0.3
    # create dataset list
    full_dataset = []
    for sample in graph_dict['train']:
        full_dataset.append(graph_dict['train'][sample]['graph'].dataset[0])
    for sample in graph_dict['test']:
        full_dataset.append(graph_dict['test'][sample]['graph'].dataset[0])
    # !!!
    # normalise, column wise for entire dataset
    all_features = torch.cat([data.x for data in full_dataset], dim=0)
    scaler = MinMaxScaler()
    scaler.fit(all_features.numpy())

    # Apply normalization to each graph
    for data in full_dataset:
        data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)
    # !!!
    # return full_dataset

    # Create DataLoaders for train and test set
    train_loader,test_loader,val_loader, dataset_dict = gcn_model.load(dataset=full_dataset,
                                        batch_size=batch_size,
                                        shuffle_dataset=False,
                                        train_split = train_split,
                                        test_split= test_split,
                                        val_split = 0)
        
    # Set up GCN model
    model = gcn_model.GCN(
        input_channels= num_node_features,
        hidden_channels= hidden_channels,
        output_channels= output_channels,
        p=dropout
        )
    
    if torch.cuda.is_available():
        # print('Using CUDA (all available GPUs)')
        print('Using CUDA')
        
        # model = DistributedDataParallel(model)
        model = model.to('cuda')
        
    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict= dataset_dict

    # Define optimizer and loss function
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=wd
        )
    
    #* Try lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.5,
        patience=10,
        verbose=True
        ) if lr_scheduling else None
    
    if output_channels == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Set up trainer
    gcntrainer = gcn_model.GCNTrainer(model=model,
                                loss_func=criterion,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                scheduler=scheduler,
                                output_dim=output_channels
                                )
    
    if wandb_params['use_wandb']:

        wandb.init(
            # set the wandb project where this run will be logged
            project = wandb_params['wandb_project'],
            name = wandb_params['wandb_name'],
            
            # track hyperparameters and run metadata
            config={
            "num_node_features": num_node_features,
            "hidden_channels": hidden_channels,
            "learning_rate": learning_rate,
            "weight_decay": wd,
            "cutoff_distance": cutoff_distance,
            "self_loops": self_loops,
            # "n_res": 1,
            # "n_sus": 1,
            "n_samples": wandb_params['n_samples'],
            "batch_size": batch_size,
            "epochs": epochs,
            }
        )
    
    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(epochs=epochs,
                                                            use_wandb=wandb_params['use_wandb'] or wandb_params['sweep'],
                                                            early_stop={
                                                                'patience': 30, 
                                                                'min_delta': 0
                                                                } if early_stop else False
                                                            )
    
    # return model, train_acc, test_acc, train_loss, test_loss
    return model