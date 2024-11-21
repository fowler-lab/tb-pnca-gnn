import sbmlsim
from src.protein_graph import ProteinGraph, pncaGraph
import wandb
import src.gcn_model as gcn_model
import torch
import pandas as pd

from typing import Union

def simple_gcn(
    batch: sbmlsim.Batch,
    n_samples,
    prop_resistant,
    n_res,
    n_sus,
    self_loops,
    cutoff_distance,
    wt_seq,
    edge_weight_func,
    batch_size,
    num_node_features,
    hidden_channels,
    learning_rate,
    wd,
    epochs,
    use_wandb,
    normalise_ews: bool = False,
    wandb_project = None,
    wandb_name = None,
    ):
    """
    Run full pipeline for BlaC GCN model.
    Includes sample dataset generation, so takes in an sbmlsim.Batch instance as an argument.

    Args:
        batch (sbmlsim.Batch): _description_
        n_samples (_type_): _description_
        prop_resistant (_type_): _description_
        n_res (_type_): _description_
        n_sus (_type_): _description_
        self_loops (_type_): _description_
        cutoff_distance (_type_): _description_
        wt_seq (_type_): _description_
        edge_weights (_type_): "dist", "1-(dist/cutoff)", "1/dist", "none"
        batch_size (_type_): _description_
        num_node_features (_type_): _description_
        hidden_channels (_type_): _description_
        learning_rate (_type_): _description_
        wd (_type_): _description_
        epochs (_type_): _description_
        use_wandb (_type_): _description_
        normalise_ews (bool, optional): _description_. Defaults to False.
        wandb_project (_type_, optional): _description_. Defaults to None.
        wandb_name (_type_, optional): _description_. Defaults to None.

    Returns:
        model (torch.nn.Module): Trained GCN model
        train_acc (list): List of training accuracies
        test_acc (list): List of test accuracies
        train_loss (list): List of training losses
        test_loss (list): List of test losses
    """
    
    if use_wandb:
        assert wandb_project is not None, "Please provide a wandb project name"
        assert wandb_name is not None, "Please provide a wandb run name"
    
    # Create samples
    sequences,mutations = batch.generate(n_samples=n_samples,
                                     proportion_resistant=prop_resistant, 
                                     n_res= n_res, 
                                     n_sus= n_sus)
    
    sequences.rename(columns = {'blaC':'allele'}, inplace=True)
    
    # Create graph
    blac = ProteinGraph(pdb = '../pdb/6h2c.pdb',
                          lig_resname= 'ISS',
                          self_loops=self_loops,
                          cutoff_distance= cutoff_distance)
    
    # Generate dataset
    blac.gen_dataset(wt_seq=wt_seq,
                 sequences=sequences,
                 edge_weights=edge_weight_func,
                 normalise=normalise_ews
                 )

    # Create DataLoaders for train and test set
    
    train_loader,test_loader,val_loader, dataset_dict = gcn_model.load(dataset=blac.dataset,
                                        batch_size=batch_size,
                                        shuffle_dataset=True,
                                        train_split = 0.8,
                                        test_split= 0.2,
                                        val_split = 0)
    
    # Set up GCN model
    model = gcn_model.GCN(
        input_channels= num_node_features,
        hidden_channels= hidden_channels
        )

    model.sequences = sequences
    model.mutations = mutations
    
    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict= dataset_dict

    # Define optimizer and loss function
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
    
    
    # Set up wandb
    wandb_project = wandb_project
    wandb_name = wandb_name

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project = wandb_project,
            name = wandb_name,
            
            # track hyperparameters and run metadata
            config={
            "num_node_features": num_node_features,
            "learning_rate": learning_rate,
            "weight_decay": wd,
            "architecture": "gCNN",
            "dataset": "blaC",
            "cutoff_distance": cutoff_distance,
            "resistant_mutations": "4angs from active site residues",
            "self_loops": self_loops,
            "n_res": n_res,
            "n_sus": n_sus,
            "n_samples": n_samples,
            "batch_size": 64,
            "epochs": epochs,
            }
        )
        
    # Train model
    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(epochs=epochs,
                                                            use_wandb=use_wandb)
    
    return model, train_acc, test_acc, train_loss, test_loss
    
def split_set_muts_gcn(
    train_batch: sbmlsim.Batch,
    test_batch: sbmlsim.Batch,
    n_samples,
    prop_resistant,
    n_res,
    n_sus,
    self_loops,
    cutoff_distance,
    wt_seq,
    edge_weight_func,
    batch_size,
    num_node_features,
    hidden_channels,
    learning_rate,
    wd,
    epochs,
    use_wandb,
    normalise_ews: bool = False,
    wandb_project = None,
    wandb_name = None,
    ):
    """
    Run full pipeline for BlaC GCN model where two distinct sample datasets are generated for the train and test set.
    Includes sample train and test dataset generation, so takes in two sbmlsim.Batch instances as arguments.

    Args:
        train_batch (sbmlsim.Batch): _description_
        test_batch (sbmlsim.Batch): _description_
        n_samples (_type_): _description_
        prop_resistant (_type_): _description_
        n_res (_type_): _description_
        n_sus (_type_): _description_
        self_loops (_type_): _description_
        cutoff_distance (_type_): _description_
        wt_seq (_type_): _description_
        edge_weight_func (_type_): _description_
        batch_size (_type_): _description_
        num_node_features (_type_): _description_
        hidden_channels (_type_): _description_
        learning_rate (_type_): _description_
        wd (_type_): _description_
        epochs (_type_): _description_
        use_wandb (_type_): _description_
        normalise_ews (bool, optional): _description_. Defaults to False.
        wandb_project (_type_, optional): _description_. Defaults to None.
        wandb_name (_type_, optional): _description_. Defaults to None.

    Returns:
        model (torch.nn.Module): Trained GCN model
        train_acc (list): List of training accuracies
        test_acc (list): List of test accuracies
        train_loss (list): List of training losses
        test_loss (list): List of test losses
    """
    
    if use_wandb:
        assert wandb_project is not None, "Please provide a wandb project name"
        assert wandb_name is not None, "Please provide a wandb run name"
    
    # Create samples
    train_n_samples = int(n_samples * 0.8)
    test_n_samples = int(n_samples * 0.2)
    
    train_sequences,train_mutations = train_batch.generate(n_samples=train_n_samples,
                                     proportion_resistant=prop_resistant, 
                                     n_res= n_res, 
                                     n_sus= n_sus)

    test_sequences,test_mutations = test_batch.generate(n_samples=test_n_samples,
                                     proportion_resistant=prop_resistant, 
                                     n_res= n_res, 
                                     n_sus= n_sus)
    
    train_sequences.rename(columns = {'blaC':'allele'}, inplace=True)
    test_sequences.rename(columns = {'blaC':'allele'}, inplace=True)
    
    # Create graph
    blac = ProteinGraph(pdb = '../pdb/6h2c.pdb',
                          lig_resname= 'ISS',
                          self_loops=self_loops,
                          cutoff_distance= cutoff_distance)
    
    # Generate dataset (for both train and test dataset)
    blac.gen_dataset(wt_seq=wt_seq,
                 sequences=train_sequences,
                 edge_weights=edge_weight_func,
                 normalise=normalise_ews
                 )
    train_dataset = blac.dataset
    
    blac.gen_dataset(wt_seq=wt_seq,
                 sequences=test_sequences,
                 edge_weights=edge_weight_func,
                 normalise=normalise_ews
                 )
    test_dataset = blac.dataset
    full_dataset = train_dataset + test_dataset

    # Create DataLoaders for train and test set
    train_loader,test_loader,val_loader, dataset_dict = gcn_model.load(dataset=full_dataset,
                                        batch_size=batch_size,
                                        shuffle_dataset=False,
                                        train_split = 0.8,
                                        test_split= 0.2,
                                        val_split = 0)

    # Set up GCN model
    model = gcn_model.GCN(
        input_channels= num_node_features,
        hidden_channels= hidden_channels
        )
    
    
    model.train_sequences = train_sequences
    model.train_mutations = train_mutations
    model.test_sequences = test_sequences
    model.test_mutations = test_mutations
    
    model.train_loader = train_loader
    model.test_loader = test_loader
    model.dataset_dict= dataset_dict

    # Define optimizer and loss function
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
    
    
    # Set up wandb
    wandb_project = wandb_project
    wandb_name = wandb_name

    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project = wandb_project,
            name = wandb_name,
            
            # track hyperparameters and run metadata
            config={
            "num_node_features": num_node_features,
            "learning_rate": learning_rate,
            "weight_decay": wd,
            "architecture": "gCNN",
            "dataset": "blaC",
            "cutoff_distance": cutoff_distance,
            "resistant_mutations": "4angs from active site residues",
            "self_loops": self_loops,
            "n_res": n_res,
            "n_sus": n_sus,
            "n_samples": n_samples,
            "batch_size": 64,
            "epochs": epochs,
            }
        )
        
    # Train model
    train_acc, test_acc, train_loss, test_loss = gcntrainer.run(epochs=epochs,
                                                            use_wandb=use_wandb,
                                                            early_stop={
                                                                'patience': 30, 
                                                                'min_delta': -0.001
                                                                })
    
    return model, train_acc, test_acc, train_loss, test_loss

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
    