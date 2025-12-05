"""
PyTorch Geometric Dataset for Protein Atom Exposure Prediction
"""
import os
import pickle
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np


class ProteinAtomDataset(Dataset):
    """
    Dataset class for loading protein graphs with atom exposure labels.

    Args:
        root (str): Root directory containing the dataset
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (callable, optional): Transform to apply to each graph
        pre_transform (callable, optional): Transform to apply before saving
        pre_filter (callable, optional): Filter to apply before saving
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.split = split
        self.root = root

        # Load protein list and depth indexes
        self.protein_df = pd.read_csv(os.path.join(root, 'protein_sample_5000.csv'))

        with open(os.path.join(root, 'depth_indexes.pkl'), 'rb') as f:
            self.depth_indexes = pickle.load(f)

        # Split dataset (80% train, 10% val, 10% test)
        n_samples = len(self.protein_df)
        train_end = int(0.8 * n_samples)
        val_end = int(0.9 * n_samples)

        if split == 'train':
            self.protein_df = self.protein_df.iloc[:train_end]
        elif split == 'val':
            self.protein_df = self.protein_df.iloc[train_end:val_end]
        elif split == 'test':
            self.protein_df = self.protein_df.iloc[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', 'test'")

        self.protein_ids = self.protein_df['pdb_id'].tolist()

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names."""
        return ['protein_sample_5000.csv', 'depth_indexes.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        """List of processed file names."""
        return [f'{pdb_id}.pt' for pdb_id in self.protein_ids]

    def download(self):
        """Download dataset (not needed as data is already present)."""
        pass

    def process(self):
        """Process raw data into PyTorch Geometric Data objects."""
        for pdb_id in self.protein_ids:
            data = self._load_protein_graph(pdb_id)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'{pdb_id}.pt'))

    def len(self) -> int:
        """Return number of samples in dataset."""
        return len(self.protein_ids)

    def get(self, idx: int) -> Data:
        """
        Get a single protein graph.

        Args:
            idx (int): Index of sample

        Returns:
            Data: PyTorch Geometric Data object
        """
        pdb_id = self.protein_ids[idx]

        # Try to load processed data
        processed_path = os.path.join(self.processed_dir, f'{pdb_id}.pt')
        if os.path.exists(processed_path):
            data = torch.load(processed_path)
        else:
            data = self._load_protein_graph(pdb_id)

        return data

    def _load_protein_graph(self, pdb_id: str) -> Data:
        """
        Load a protein graph from CSV files.

        Args:
            pdb_id (str): Protein PDB ID

        Returns:
            Data: PyTorch Geometric Data object
        """
        protein_dir = os.path.join(self.root, 'sadic_data', pdb_id)

        # Load nodes
        nodes_df = pd.read_csv(
            os.path.join(protein_dir, f'{pdb_id}__graphein__ATOM_nodes.csv'),
            index_col=0
        )

        # Load edges
        edges_df = pd.read_csv(
            os.path.join(protein_dir, f'{pdb_id}__graphein__ATOM_edges.csv'),
            index_col=0
        )

        # Extract node features (skip first few columns that are identifiers)
        # We'll use numeric features: b_factor, coords, and all biochemical properties
        feature_cols = nodes_df.columns[6:]  # Skip identifiers
        x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)

        # Create node ID to index mapping
        node_ids = nodes_df['original_index'].values
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Create edge index
        edge_index = []
        edge_attr = []

        for _, edge in edges_df.iterrows():
            src = node_to_idx.get(edge['idx_0'])
            dst = node_to_idx.get(edge['idx_1'])

            if src is not None and dst is not None:
                edge_index.append([src, dst])
                # Add edge attributes (distance)
                distance = edge['distance'] if not pd.isna(edge['distance']) else 0.0
                edge_attr.append([distance])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Load target labels (atom exposure/depth)
        if pdb_id in self.depth_indexes:
            y = torch.tensor(self.depth_indexes[pdb_id], dtype=torch.float)
        else:
            # If no labels, create dummy labels
            y = torch.zeros(len(nodes_df), dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pdb_id=pdb_id,
            num_nodes=len(nodes_df)
        )

        return data


if __name__ == '__main__':
    # Test dataset loading
    dataset = ProteinAtomDataset(root='dataset/', split='train')
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample graph:")
    print(f"  PDB ID: {sample.pdb_id}")
    print(f"  Number of atoms: {sample.num_nodes}")
    print(f"  Number of edges: {sample.num_edges}")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge features shape: {sample.edge_attr.shape}")
    print(f"  Target shape: {sample.y.shape}")
