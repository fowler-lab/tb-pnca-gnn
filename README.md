# tb-pnca-gnn

A graph convolutional neural network (GCN) for predicting Pyrazinamide (PZA) resistance in *Mycobacterium tuberculosis* based on mutations in the pncA gene.

## Overview

This project implements a Graph Convolutional Network to predict antimicrobial resistance by modeling protein structures as graphs. The model uses protein structure graphs and node features including amino acid properties and structural features.

## Model Architecture

The GCN uses:
- **3 GraphConv layers** with batch normalization
- **Global mean pooling** for graph-level predictions
- **Dropout** for regularization
- **Cross-entropy loss** for binary classification

## Results

The model achieves:
- **F1 Score**: 0.816 on Test dataset
- **Binary classification** of pyrazinamide resistance/susceptibility
- **Interpretable predictions** through GNN explainability
