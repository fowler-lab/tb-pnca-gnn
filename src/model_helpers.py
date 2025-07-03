import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
def convert_3letter_to_sequence(resnames):
    """Converts array of 3-letter amino acid names to allele sequence of 1-letter amino acids.

    Args:
        resnames (array): mdanalysis.Universe.nodes.residues.resnames. 
                        eg. for a pnca graph object: pnca.nodes.residues.resnames
    """
    
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
        'TRP': 'W', 'TYR': 'Y'}
    
    sequence = ''.join([aa_dict[res] for res in resnames])
    
    return sequence

def redefine_graph(graph_dict, 
                   cutoff_distance, 
                   edge_weight_func, 
                   normalise_ews, 
                   lambda_param=None,
                   no_node_mpfs=False,
                   no_node_chem_feats=False,
                   rand_node_feats=False,
                   shuffle_edges=False):
    
    assert sum([no_node_mpfs, no_node_chem_feats, rand_node_feats]) <= 1, \
        "Can only set one of no_node_mpfs, no_node_chem_feats, or rand_node_feats to be True"

    print(f'Adjusting edge index and attaching edge weights for cutoff distance {cutoff_distance}')

    if shuffle_edges:
        print('Shuffling edges')
        
    if no_node_mpfs:
        print('Removing metapredictor features from node features')
    
    if no_node_chem_feats:
        print('Removing sbmlcore features from node features')
    
    if rand_node_feats:
        print('Randomising node features')
        
    for sample_set in graph_dict:
        for sample in graph_dict[sample_set]:
            graph = graph_dict[sample_set][sample]['graph']
            
            # assert graph.dataset[0].x.size() == torch.Size([185, 16]), \
            assert graph.dataset[0].x.size(1) == 16, \
                "Reload graph_dict"
                
            # reset graph attributes based on new cutoff distance
            graph.cutoff_distance = cutoff_distance
            
            # calc new edge index and edge weights
            edge_index, d_array = graph._get_protein_struct_edges(graph.nodes.center_of_mass(compound='residues'))
            edge_dists = graph._gen_edge_dists(edge_index, d_array)
            edge_attr = graph.calc_edge_weights(edge_weight_func, edge_dists, lambda_param)
            
            if normalise_ews:
                edge_attr = graph.process_edge_weights(edge_attr)
            
            if shuffle_edges:
                # shuffle edges
                row, col = edge_index
                perm = torch.randperm(row.size(0))
                edge_index = torch.stack([row, col[perm]], dim=0)
                edge_attr = edge_attr[torch.randperm(len(edge_attr))]
            
            if no_node_mpfs:
                # remove metapredictor features
                raise NotImplementedError("Need to check which index of tensor to remove")
                new_node_feats = graph.dataset[0].x[:, :12]
                graph_dict[sample_set][sample]['graph'].dataset[0].x = new_node_feats
                
            if no_node_chem_feats:
                # remove sbmlcore features
                raise NotImplementedError("Need to check which index of tensor to remove")
                new_node_feats = graph.dataset[0].x[:, 12:]
                graph_dict[sample_set][sample]['graph'].dataset[0].x = new_node_feats
                
            if rand_node_feats:
                # replace node features with random values
                num_nodes = graph.dataset[0].x.size(0)
                num_features = graph.dataset[0].x.size(1)
                new_node_feats = torch.rand((num_nodes, num_features))
                graph_dict[sample_set][sample]['graph'].dataset[0].x = new_node_feats
                
            # change edge index and edge weights for Data object
            graph_dict[sample_set][sample]['graph'].dataset[0].edge_index = edge_index
            graph_dict[sample_set][sample]['graph'].dataset[0].edge_attr = edge_attr
    