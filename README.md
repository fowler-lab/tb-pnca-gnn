# tb-pnca-gnn

A graph convolutional neural network (GCN) for predicting Pyrazinamide (PZA) resistance in *Mycobacterium tuberculosis* based on mutations in the pncA gene.

## Overview

This project implements a Graph Convolutional Network to predict antimicrobial resistance by modeling protein structures as graphs. The model uses protein structure graphs and node features including amino acid properties and structural features.

# tb-pnca-gnn

A Graph Convolutional Network (GCN) for predicting Pyrazinamide (PZA) resistance in *Mycobacterium tuberculosis* from mutations in the *pncA* gene.

## Overview

Pyrazinamide is a critical first-line antibiotic for tuberculosis treatment, but resistance prediction remains challenging due to the diverse range of mutations in the *pncA* gene. This project implements a GCN that models the PncA protein structure as a graph, combining structural information with amino acid features to predict resistance phenotypes.

The model was trained on phenotypically characterised isolates and achieves an **F1 score of 0.816** on the held-out test set.

## Installation

```bash
git clone https://github.com/fowler-lab/tb-pnca-gnn.git
cd tb-pnca-gnn
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- Biopython
- pandas, numpy, scikit-learn
- Weights & Biases (optional, for experiment tracking)

## Repository Structure

```
tb-pnca-gnn/
├── src/                    # Core source code
│   ├── gcn_model.py        # GCN architecture
│   ├── protein_graph.py    # Protein graph construction
│   ├── evaluation.py       # Model evaluation utilities
│   └── model_helpers.py    # Training helpers
├── train/                  # Training scripts
│   ├── train.py            # Main training script
│   └── configs/            # YAML configuration files
├── notebooks/              # Analysis notebooks (follow paper workflow)
│   ├── 01_create_sequences.ipynb
│   ├── 02_create_graph_dataset.ipynb
│   ├── 03_train_model.ipynb
│   ├── 04_bootstrapping.ipynb
│   ├── 05_gnn_explainer.ipynb
│   └── ...
├── data/                   # Datasets and features
├── saved_models/           # Pre-trained model weights
└── requirements.txt
```

## Usage

### Training a Model

Using the training script:

```bash
cd train
python train.py
```

Configuration can be modified in `train/configs/config.yaml`.

### Using Pre-trained Models

Pre-trained models are available in `saved_models/`:

| Model | F1 Score | Description |
|-------|----------|-------------|
| `full_model/F1=0.816_epoch=1119.pth` | 0.816 | Full model with all features |
| `codon-split/F1=0.835_epoch=246.pth` | 0.835 | Codon-based train/test split |
| `cluster-split/F1=0.798_epoch=248.pth` | 0.798 | Cluster-based train/test split |

### Notebooks

The notebooks in `notebooks/` follow the analysis pipeline described in the paper:

1. **01_create_sequences.ipynb** - Process mutation data and create sequences
2. **02_create_graph_dataset.ipynb** - Build protein graphs with node features
3. **03_train_model.ipynb** - Train and evaluate the GCN
4. **04_bootstrapping.ipynb** - Bootstrap confidence intervals
5. **05_gnn_explainer.ipynb** - Model interpretability analysis
6. **06_node_feature_importance.ipynb** - Feature importance analysis

## Data

Relevant data files are included in the `data/` directory, including:
- Training and test sequences
- Protein structure features (MAPP scores, SNAP2 predictions, ΔΔG values)
- Pre-computed graph datasets in `data/inputs/`

## Citation

If you use this code, please cite:

> Dissanayake, D. et al. (2025). Predicting pyrazinamide resistance in Mycobacterium tuberculosis using a graph convolutional network. *bioRxiv*. https://doi.org/10.1101/2025.10.28.685176

```bibtex
@article{dissanayake2025pnca,
  title={Predicting pyrazinamide resistance in Mycobacterium tuberculosis using a graph convolutional network},
  author={Dissanayake, Dylan and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.10.28.685176}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Dylan Dissanayake - [dylan.dissanayake@msdtc.ox.ac.uk](mailto:dylan.dissanayake@msdtc.ox.ac.uk)

Philip W Fowelr - [philip.fowler@ndm.ox.ac.uk](mailto:philip.fowler@ndm.ox.ac.uk)

Fowler Lab, University of Oxford