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