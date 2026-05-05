# DeepHeptox

**An Interpretable Deep Learning Model for Multi-Endpoint Hepatotoxicity Prediction of Chemical Compounds**

DeepHeptox uses a Graph Attention Network (GAT) to predict hepatotoxicity from molecular SMILES strings. Atoms are encoded as node features and molecular topology is used as the graph structure.

\---

## Project Structure

```
.
├── train\\\_test.py       # Main training \\\\\\\\\\\\\\\& test evaluation script
├── 5cv.py              # 5-fold cross-validation script
├── model.py            # GATNet model definition
├── dataset.py          # MolDataset: loads CSV and builds graphs
├── featurizer.py       # Atom-level molecular featurization
├── evaluate.py         # Evaluation metrics (ACC, AUC, SE, SP, F1)
├── seed.py             # Reproducibility utility
├── utils.py            # One-hot encoding helper
├── train\\\_hepatitis.csv # Training data (SMILES + label)
└── test\\\_hepatitis.csv  # Test data (SMILES + label)
```

\---

## Requirements

```bash
pip install torch torch-geometric rdkit-pypi scikit-learn pandas tqdm scipy
```

> Make sure your PyTorch and PyG versions are compatible. Refer to the \\\\\\\\\\\\\\\[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA version.

\---

## Data Format

Both `train\\\_hepatitis.csv` and `test\\\_hepatitis.csv` must contain the following two columns:

|smiles|label|
|-|-|
|CC(=O)Oc1ccccc1C(=O)O|1|
|c1ccccc1|0|

* `smiles`: canonical SMILES string of the molecule
* `label`: binary label (`1` = hepatotoxic, `0` = non-hepatotoxic)

\---

## Training \& Testing

Run `train\\\\\\\\\\\\\\\_test.py` to train the model on `train\\\\\\\\\\\\\\\_hepatitis.csv` and evaluate on `test\\\\\\\\\\\\\\\_hepatitis.csv`:

```bash
python train\\\\\\\\\\\\\\\_test.py
```

