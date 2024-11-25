import sbmlsim
from src.protein_graph import pncaGraph
import wandb
import src.gcn_model as gcn_model
import torch
import pandas as pd

from typing import Union

def pnca_simpleGCN(
    sequences: Union[pd.DataFrame, dict],
    self_loops,
    cutoff_distance,
    edge_weight_func,
    batch_size,
    num_node_features,
    hidden_channels,
    learning_rate,
    wd,
    epochs,
    normalise_ews: bool = True,
    wandb_params = {'use_wandb': False, 'wandb_project': None, 'wandb_name': None}
    ):
    """
    Runs PncA GCN model pipeline. Datasets must be generated prior.

    Args:
        sequences (Union[pd.DataFrame, dict]): If train/test split already done, provide dict in form of {'train': train_df, 'test': test_df}
        self_loops (_type_): _description_
        cutoff_distance (_type_): _description_
        edge_weight_func (_type_): _description_
        batch_size (_type_): _description_
        num_node_features (_type_): _description_
        hidden_channels (_type_): _description_
        learning_rate (_type_): _description_
        wd (_type_): _description_
        epochs (_type_): _description_
        normalise_ews (bool, optional): _description_. Defaults to True.

    Returns:
        model (torch.nn.Module): Trained GCN model
        train_acc (list): List of training accuracies
        test_acc (list): List of test accuracies
        train_loss (list): List of training losses
        test_loss (list): List of test losses
    """
    
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
        hidden_channels= hidden_channels
        )
    
    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict= dataset_dict

    # Define optimizer and loss function
    #* Try AdamW optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=wd
        )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Set up trainer
    gcntrainer = gcn_model.GCNTrainer(model=model,
                                loss_func=criterion,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                test_loader=test_loader)
    
    if wandb_params['use_wandb']:

        wandb.init(
            # set the wandb project where this run will be logged
            project = wandb_params['wandb_project'],
            name = wandb_params['wandb_name'],
            
            # track hyperparameters and run metadata
            config={
            "num_node_features": num_node_features,
            "learning_rate": learning_rate,
            "weight_decay": wd,
            "architecture": "gCNN",
            "dataset": "blaC",
            "cutoff_distance": cutoff_distance,
            "self_loops": self_loops,
            "n_res": 1,
            "n_sus": 1,
            "n_samples": wandb_params['n_samples'],
            "batch_size": 64,
            "epochs": epochs,
            }
        )
    
    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(epochs=epochs,
                                                            use_wandb=wandb_params['use_wandb'],
                                                            early_stop=False
                                                            )
    
    return model, train_acc, test_acc, train_loss, test_loss
    