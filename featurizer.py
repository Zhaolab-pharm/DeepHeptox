from rdkit import Chem
from scipy.sparse import coo_matrix
from utils import one_hot_encoding
import torch

class MoleculeFeaturizer:
    def __init__(self):
        pass

    def _atom_featurizer(self, atom):
        nums = [5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 33, 35, 51, 53]
        f1 = one_hot_encoding(atom.GetAtomicNum(), nums)
        f2 = one_hot_encoding(atom.GetTotalDegree(), list(range(5)))
        f3 = one_hot_encoding(int(atom.GetHybridization()), list(range(len(Chem.HybridizationType.names) - 1)))
        f4 = one_hot_encoding(int(atom.GetChiralTag()), list(range(len(Chem.ChiralType.names) - 1)))
        f5 = one_hot_encoding(atom.GetTotalNumHs(), list(range(5)))
        f6 = [1 if atom.GetIsAromatic() else 0]
        return f1 + f2 + f3 + f4 + f5 + f6

    def __call__(self, mol):
        if mol is None:
            raise RuntimeError("failed")
        atoms = [self._atom_featurizer(a) for a in mol.GetAtoms()]

        adj = Chem.GetAdjacencyMatrix(mol)
        coo = coo_matrix(adj)
        edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

        return {
            "x": atoms,
            "edge_index": edge_index
        }