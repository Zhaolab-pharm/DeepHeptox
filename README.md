# DeepHeptox

**An Interpretable Deep Learning Model for Multi-Endpoint Hepatotoxicity Prediction of Chemical Compounds**

## Environment Setup
```bash
conda create -n DeepHeptox python=3.13 -y
```

```bash
conda activate DeepHeptox
```

```bash
conda install pandas=2.3.0
```

```bash
pip install torch==2.8.0```
```

```bash
pip install torch-scatter==2.1.2 \
            torch-sparse==0.6.18 \
            torch-cluster==1.6.3 \
            torch-spline-conv==1.2.2 \
            torch-geometric==2.6.1
```

```bash
conda install rdkit=2025.03.6
```

```bash
conda install scikit-learn=1.7.2
```

## Project Structure

```
.
├── train_test.py       # Main training& test evaluation script
├── 5cv.py              # 5-fold cross-validation script
├── model.py            # GATNet model definition
├── dataset.py          # MolDataset: loads CSV and builds graphs
├── featurizer.py       # Atom-level molecular featurization
├── evaluate.py         # Evaluation metrics
├── seed.py             # Reproducibility utility
├── utils.py            # One-hot encoding helper
├── train_hepatitis.csv # Training data (SMILES + label)
└── test_hepatitis.csv  # Test data (SMILES + label)
```

\---

## Requirements

```bash
pip install torch torch-geometric rdkit-pypi scikit-learn pandas tqdm scipy
```
Our model is implemented using PyTorch 2.8.0 and PyTorch Geometric 2.6.1, and trained on a single NVIDIA GPU.
\---

## Data Format

Both `train_hepatitis.csv` and `test_hepatitis.csv` must contain the following two columns:

|smiles|label|
|-|-|
|CC(=O)Oc1ccccc1C(=O)O|1|
|c1ccccc1|0|

* `smiles`: canonical SMILES string of the molecule
* `label`: binary label (`1` = positive, `0` = negative)

\---

## Training \& Testing

Run `train_test.py` to train the model on `train_hepatitis.csv` and evaluate on `test_hepatitis.csv`:

```bash
python train_test.py
```

