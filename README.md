# tb-pnca-gnn

A graph convolutional neural network (GCN) for predicting Pyrazinamide (PZA) resistance in *Mycobacterium tuberculosis*.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citing](#citing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository contains code and data for training and evaluating a graph convolutional neural network (GCN) to predict resistance to the antibiotic Pyrazinamide (PZA) in *Mycobacterium tuberculosis* using protein structure and mutation data.

## Features

- GCN-based model for resistance prediction
- Protein structure and mutation feature integration
- Model training, evaluation, and explainability tools
- Preprocessed datasets and scripts for reproducibility

## Project Structure

```
data/                # Datasets and features
dd_pnca/             # Jupyter notebooks for analysis and experiments
pdb/                 # Protein structure files
src/                 # Source code (models, helpers, etc.)
requirements.txt     # Python dependencies
README.md            # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tb-pnca-gnn.git
   cd tb-pnca-gnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Model

To train or evaluate the model, use the scripts in the `src/` directory. For example:

```bash
python src/run_model.py --help
```

Or use the provided Jupyter notebooks in `dd_pnca/` for interactive analysis.

### Example

```python
from src import run_model

# Example usage (update with actual function calls)
run_model.train(...)
run_model.evaluate(...)
```

## Data

- Datasets are provided in the `data/` directory.
- Protein structures are in `pdb/`.
- See the notebooks in `dd_pnca/` for data exploration and preprocessing steps.

## Training

To train the model, use:

```bash
python src/run_model.py --train --config configs/train_config.yaml
```

(Adjust the command based on your actual CLI/API.)

## Evaluation

To evaluate a trained model:

```bash
python src/run_model.py --eval --model-path path/to/model.pth
```

## Results

- Model checkpoints and results are saved in `dd_pnca/../saved_models/`.
- See the `dd_pnca/plot_wandb_curves.ipynb` notebook for performance plots.

