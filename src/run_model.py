import sbmlsim
from src.protein_graph import pncaGraph
import wandb
import src.gcn_model as gcn_model
import torch
import pandas as pd
from torch_geometric.data import Data

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
    
    if dataset is None:
        
        wt_seq = 'MRALIIVDVQNDFCEGGSLAVTGGAALARAISDYLAEAADYHHVVATKDFHIDPGDHFSGTPDYSSSWPPHCVSGTPGADFHPSLDTSAIEAVFYKGAYTGAYSGFEGVDENGTPLLNWLRQRGVDEVDVVGIATDHCVRQTAEDAVRNGLATRVLVDLTAGVSADTTVAALEEMRTASVELVCS'
        
        # Create graph
        pnca = pncaGraph(pdb='../pdb/3PL1-PZA.pdb',
                        lig_resname='PZA', 
                        self_loops=self_loops,
                        cutoff_distance=cutoff_distance)
        
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
        output_channels= output_channels
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
    
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=learning_rate, 
    #     weight_decay=wd
    #     )
    
    #* Try AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=wd
        )
    
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
    