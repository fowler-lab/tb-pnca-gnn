"""
SE(3)-equivariant residue‑graph classifier (binary‑as‑2‑class)
using lucidrains/se3-transformer-pytorch, with per‑residue Atom14
(14×3) coordinates + masks.

Nodes = amino‑acid residues.
• Node position = CA (fallback: mean of N/CA/C/O if CA absent).
• Vector (type‑1) features = relative vectors from node position to all 14 atoms.
• Scalar (type‑0) features = residue one‑hot + atom‑exist mask + optional chem features.
• Edges = Fourier‑encoded CA–CA distances and sequence offsets; sparse neighbor graph.
• Output = 2 logits (treat binary as multiclass).

Atom14 background: this is the compact, fixed‑width heavy‑atom representation
that AlphaFold/OpenFold use (backbone N,CA,C,O + up to 10 side‑chain atoms),
exposed together with masks and mappings (atom14↔atom37) in residue_constants
and all_atom helpers. See:
- OpenFold residue_constants.py (masks & maps)
- AlphaFold all_atom.py (atom14 utilities)
"""
from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.utils import fourier_encode

# =============================
# Constants / small utilities
# =============================

ATOM14 = 14
BB_IDX = {"N": 0, "CA": 1, "C": 2, "O": 3}  # standard Atom14 ordering
CB_IDX = 4  # conventional slot for CB when present

# Fourier encoding helper for edges
class EdgeFeaturizer(nn.Module):
    def __init__(self, num_encodings: int = 8, include_self: bool = True):
        super().__init__()
        self.num_encodings = num_encodings
        self.include_self = include_self

    def forward(self, dists: torch.Tensor, seq_offsets: torch.Tensor) -> torch.Tensor:
        """dists: (B, L, L, 1) in Å; seq_offsets: (B, L, L, 1) absolute |i-j|.
        Returns edges (B, L, L, E), where E = 2 * (2 * num_encodings + (1 if include_self else 0)).
        """
        d_enc = fourier_encode(dists, self.num_encodings, include_self=self.include_self)  # (B,L,L, 2*(n+1))
        s_enc = fourier_encode(seq_offsets, self.num_encodings, include_self=self.include_self)
        return torch.cat((d_enc, s_enc), dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(x.dtype)
    x = x * mask
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return x.sum(dim=dim) / denom

# =============================
# Data containers
# =============================

@dataclass
class ResidueGraphItem:
    # Required per‑residue tensors
    aatype: torch.Tensor            # (L,) int in [0..20] with 20 AA + unknown
    atom14_pos: torch.Tensor        # (L, 14, 3) Å
    atom14_mask: torch.Tensor       # (L, 14) bool
    # Optional extras
    residue_chem: Optional[torch.Tensor] = None  # (L, Fchem) scalars (charge, hydrop., etc.)
    label: Optional[int] = None                   # 0 or 1

class MockDataset(Dataset):
    """Replace with a real loader that builds Atom14 arrays from PDB/mmCIF using
    OpenFold/AlphaFold residue_constants mappings (atom14 masks & maps).
    
    atom37/14:
    https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/all_atom.py
        
    Args:
        n: number of samples
        L: sequence length
        Fchem: number of per‑residue chemical features (optional)
    """
    def __init__(self, n: int = 16, L: int = 128, Fchem: int = 8):
        torch.manual_seed(0)
        self.items = []
        for _ in range(n):
            aatype = torch.randint(0, 21, (L,)) # create vector of random aa's with length L
            atom14_mask = torch.zeros(L, ATOM14, dtype=torch.bool)
            # backbone always present
            atom14_mask[:, 0:4] = True
            # random side‑chain occupancy pattern (just for shape sanity)
            rand_sc = torch.rand(L, ATOM14-4) > 0.2
            atom14_mask[:, 4:] = rand_sc
            atom14_pos = torch.randn(L, ATOM14, 3) # coordinates
            residue_chem = torch.randn(L, Fchem) # aa features
            label = np.random.randint(2)
            self.items.append(ResidueGraphItem(aatype, atom14_pos, atom14_mask, residue_chem, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# =============================
# Collate: builds residue graph batch
# =============================

@dataclass
class Batch:
    features: Dict[str, torch.Tensor]  # {'0': (B,L,D0,1), '1': (B,L,D1,3)}
    coors: torch.Tensor                # (B, L, 3) node positions (CA‑centric)
    mask: torch.Tensor                 # (B, L) bool
    edges: torch.Tensor                # (B, L, L, E)
    labels: Optional[torch.Tensor]     # (B,) long


def collate(items: list[ResidueGraphItem]) -> Batch:
    B = len(items)
    L = max(it.atom14_pos.shape[0] for it in items)

    # tensors
    aatype = torch.stack([torch.nn.functional.pad(it.aatype, (0, L - it.aatype.shape[0]), value=20) for it in items])  # pad unknown
    mask = torch.stack([torch.nn.functional.pad(it.atom14_mask.any(dim=-1), (0, L - it.atom14_mask.shape[0]), value=False) for it in items])  # (B,L)
    atom14_mask = torch.stack([torch.nn.functional.pad(it.atom14_mask, (0,0,0, L - it.atom14_mask.shape[0]), value=False) for it in items])    # (B,L,14)
    atom14_pos = torch.stack([torch.nn.functional.pad(it.atom14_pos, (0,0,0,0,0, L - it.atom14_pos.shape[0])) for it in items])                # (B,L,14,3)

    # node coors = CA, fallback to mean of N/CA/C/O when CA missing
    bb_mask = atom14_mask[..., 0:4]                        # (B,L,4)
    bb_pos = atom14_pos[..., 0:4, :]                       # (B,L,4,3)
    ca_pos = atom14_pos[..., BB_IDX["CA"], :]              # (B,L,3)
    has_ca = atom14_mask[..., BB_IDX["CA"]]
    bb_mean = (bb_pos * bb_mask[..., None].float()).sum(dim=-2) / (bb_mask.float().sum(dim=-1).clamp(min=1e-6)[..., None])
    coors = torch.where(has_ca[..., None], ca_pos, bb_mean) # (B,L,3)

    # type‑1 vectors: (atom14 - node_coors)
    rel_vecs = atom14_pos - coors[..., None, :]            # (B,L,14,3)
    rel_vecs = rel_vecs * atom14_mask[..., None].float()   # zero out non‑existent atoms

    # type‑0 scalars: residue one‑hot + atom14 exist mask + optional chem features
    R = 21
    res_onehot = torch.nn.functional.one_hot(aatype.clamp(max=R-1), num_classes=R).float()  # (B,L,R)
    atom14_exist = atom14_mask.float()                                                        # (B,L,14)
    chem_feats = []
    if items[0].residue_chem is not None:
        chem_feats = [torch.stack([torch.nn.functional.pad(it.residue_chem, (0,0,0, L - it.residue_chem.shape[0])) for it in items])]  # (B,L,Fchem)
    scalars = torch.cat([res_onehot, atom14_exist] + chem_feats, dim=-1)                     # (B,L,D0)

    # package features in fiber format expected by the repo
    feats_type0 = scalars.unsqueeze(-1)                 # (B,L,D0,1)
    feats_type1 = rel_vecs                              # (B,L,14,3)
    features = {"0": feats_type0, "1": feats_type1}

    # edges: CA–CA distance + |i-j|
    ca = coors
    pair_dists = torch.cdist(ca, ca).unsqueeze(-1)      # (B,L,L,1)
    seq_idx = torch.arange(L).view(1, L, 1).expand(B, -1, L)
    seq_offsets = (seq_idx - seq_idx.transpose(1, 2)).abs().unsqueeze(-1).float()  # (B,L,L,1)
    edge_dim = 2 * (2 * 8 + 1) * 1  # matches EdgeFeaturizer with num_encodings=8, include_self=True, for two channels (dist & |i-j|)
    edge_featurizer = EdgeFeaturizer(num_encodings=8, include_self=True)
    edges = edge_featurizer(pair_dists, seq_offsets)    # (B,L,L,E)

    # labels
    labels = torch.tensor([it.label if it.label is not None else -1 for it in items], dtype=torch.long)

    return Batch(features=features, coors=coors, mask=mask, edges=edges, labels=labels)

# =============================
# Model: SE(3) Transformer + pooling + classifier
# =============================

class InputProj(nn.Module):
    def __init__(self, d0_in: int = 43, d1_in: int = 14, dim: int = 256):
        super().__init__()
        # type-0 (scalars): channel projection 43 -> 256; bias is fine
        self.proj0 = nn.Linear(d0_in, dim)
        # type-1 (vectors): channel projection 14 -> 256; NO bias, don't touch last dim=3
        self.proj1 = nn.Linear(d1_in, dim, bias=False)

    def forward(self, feats0, feats1):
        # feats0: (B, L, D0, 1) -> (B, L, dim, 1)
        f0 = self.proj0(feats0.squeeze(-1)).unsqueeze(-1)

        # feats1: (B, L, D1, 3) -> (B, L, dim, 3)
        # mix across the D1 channel dimension only; leave the xyz (last dim=3) intact
        f1 = feats1.permute(0, 1, 3, 2)   # (B, L, 3, D1)
        f1 = self.proj1(f1)               # (B, L, 3, dim)
        f1 = f1.permute(0, 1, 3, 2).contiguous()  # (B, L, dim, 3)
        return {'0': f0, '1': f1}

class ResidueSE3Classifier(nn.Module):
    def __init__(self, d0_in: int, d1_in: int, edge_dim: int, model_dim: int = 256,
                 hidden: Dict[int,int] = {0:128,1:32,2:8}):
        super().__init__()
        self.inproj = InputProj(d0_in, d1_in, model_dim)

        self.se3 = SE3Transformer(
            dim = model_dim,            # must match projector output
            depth = 4,
            num_degrees = 3,
            input_degrees = 2,
            output_degrees = 2,
            hidden_fiber_dict = hidden,
            out_fiber_dict = {0:128, 1:8},
            reduce_dim_out = False,
            edge_dim = edge_dim,
            num_neighbors = 2,
            attend_sparse_neighbors = False,
            valid_radius = 12.0,
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, batch, adj_mat=None):
        feats = self.inproj(batch.features['0'], batch.features['1'])
        out = self.se3(feats, batch.coors, batch.mask, edges=batch.edges, adj_mat=adj_mat)
        x0 = out['0']                       # (B,L,128)
        g = (x0 * batch.mask.unsqueeze(-1)).sum(1) / batch.mask.sum(1, keepdim=True).clamp(min=1)
        return self.cls(g)                  # (B,2)

# =============================
# Training sketch
# =============================

def train_step(model: ResidueSE3Classifier, batch: Batch, optimizer: torch.optim.Optimizer):
    model.train()
    logits = model(batch)
    y = batch.labels
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
    return loss.item(), acc


if __name__ == "__main__":
    # demo run with synthetic data
    ds = MockDataset(n=4, L=96)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    batch = next(iter(loader))

    d_scalar = batch.features['0'].shape[-2]
    d_vec = batch.features['1'].shape[-2]
    edge_dim = batch.edges.shape[-1]

    model = ResidueSE3Classifier(d_scalar, d_vec, edge_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    loss, acc = train_step(model, batch, opt)
    print({'loss': round(loss, 4), 'acc': round(acc, 3), 'd_scalar': d_scalar, 'd_vec': d_vec, 'edge_dim': edge_dim})

"""
Implementation notes & hooks for real data
-----------------------------------------
1) Building Atom14 from PDB:
   • Use OpenFold/AlphaFold residue_constants to map PDB atoms→atom37→atom14
     and to get the masks (atom14_atom_exists) – ensures correct ordering
     with backbone slots [N, CA, C, O] followed by side‑chains.
   • Keep per‑residue aatype (0..20, X=unknown) for the one‑hot.

2) Node choice:
   • We set node coordinates to CA (fallback to mean of N/CA/C/O when missing).
   • Degree‑1 channels carry all 14 relative vectors (atom_pos − CA), which
     preserves SE(3) equivariance for these features.

3) Edges & sparsity:
   • We encode CA–CA distance plus |i−j| with Fourier features.
   • For sparsity, pass adj_mat with sequence band neighbors (±2) OR k‑NN in 3D.

4) Binary‑as‑multiclass:
   • We return two logits. Train with CrossEntropyLoss on labels {0,1}.

5) Extendable scalars:
   • Add pLDDT/ASA/charges/hydropathy/etc. to residue_chem.

6) Gly Cβ (optional):
   • If you want Cβ‑based edges, use OpenFold's pseudo_beta_fn to synthesize
     a Gly Cβ, then compute Cβ–Cβ distances.
"""
