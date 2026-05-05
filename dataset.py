import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
from featurizer import MoleculeFeaturizer
import torch


class MolDataset:
    def __init__(self, csv_path):
        self.featurizer = MoleculeFeaturizer()
        self.data_list = []
        self._process_csv(csv_path)

    def _process_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        assert 'smiles' in df.columns and 'label' in df.columns, "CSV must contain 'smiles' and 'label'"

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            smiles = row['smiles']
            label = row['label']
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(f"Skipping invalid SMILES: {smiles}")
                continue

            features = self.featurizer(mol)

            data = Data()
            data.smiles = smiles
            data.y = torch.tensor([label], dtype=torch.float32)

            data.x = torch.tensor(features['x'], dtype=torch.float32)
            data.edge_index = features['edge_index']

            self.data_list.append(data)

        print(f"Processed {len(self.data_list)} molecules successfully.")

        if len(self.data_list) > 0:
            sample = self.data_list[0]
            print(f"Atom feature dimension: {sample.x.shape[1]}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_feature_dims(self):
        if len(self.data_list) == 0:
            return None

        sample = self.data_list[0]
        return {
            'node_feature_dim': sample.x.shape[1]
        }