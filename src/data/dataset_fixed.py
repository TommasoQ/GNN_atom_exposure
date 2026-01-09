"""
PyTorch Geometric Dataset for Protein Atom Exposure Prediction
FIXED VERSION - Phase 2
"""
import os
import pickle
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np

# Import feature engineering module
try:
    from .feature_engineering import (
        extract_all_features,
        FeatureNormalizer,
        get_feature_dimensions
    )
except ImportError:
    # For direct execution
    from feature_engineering import (
        extract_all_features,
        FeatureNormalizer,
        get_feature_dimensions
    )


class ProteinAtomDataset(Dataset):
    """
    Dataset class for loading protein graphs with atom exposure labels.

    FIXES APPLIED:
    1. depth_indexes DataFrame → dict conversion
    2. Filter proteins without labels (406 proteins excluded)
    3. Use feature engineering pipeline (numerical + categorical + geometric)
    4. Feature normalization
    5. Proper target matching

    Args:
        root (str): Root directory containing the dataset
        split (str): Dataset split - 'train', 'val', or 'test'
        normalize_features (bool): Whether to normalize numerical features
        normalizer_path (str): Path to save/load normalization statistics
        transform (callable, optional): Transform to apply to each graph
        pre_transform (callable, optional): Transform to apply before saving
        pre_filter (callable, optional): Filter to apply before saving
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        normalize_features: bool = True,
        normalizer_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.split = split
        self.root = root
        self.normalize_features = normalize_features

        # Load protein list
        protein_csv_path = os.path.join(root, 'dataset', 'protein_sample_5000.csv')
        self.protein_df = pd.read_csv(protein_csv_path)

        # FIX 1: Load depth_indexes (use pre-converted dict version for speed)
        depth_dict_path = os.path.join(root, 'dataset', 'depth_indexes_dict.pkl')
        depth_df_path = os.path.join(root, 'dataset', 'depth_indexes.pkl')

        # Try to load pre-converted dict version first (much faster!)
        if os.path.exists(depth_dict_path):
            print(f"Loading pre-converted depth_indexes dict...")
            with open(depth_dict_path, 'rb') as f:
                self.depth_indexes = pickle.load(f)
            print(f"  Loaded {len(self.depth_indexes)} proteins")
        else:
            # Fallback: load DataFrame and convert (slow)
            print(f"WARNING: depth_indexes_dict.pkl not found, converting from DataFrame (slow)...")
            print(f"  Run: python experiments/progress/preprocess_depth_indexes.py")
            with open(depth_df_path, 'rb') as f:
                depth_indexes_df = pickle.load(f)

            # Convert DataFrame to nested dict: {pdb_id: {atom_name: depth_value}}
            print(f"Converting depth_indexes DataFrame to dict...")
            if isinstance(depth_indexes_df, pd.DataFrame):
                self.depth_indexes = {}
                for pdb_id in depth_indexes_df['pdb_id'].unique():
                    pdb_data = depth_indexes_df[depth_indexes_df['pdb_id'] == pdb_id]
                    self.depth_indexes[pdb_id] = dict(zip(
                        pdb_data['atom_name'],
                        pdb_data['depth_index']
                    ))
                print(f"  Converted {len(self.depth_indexes)} proteins")
            else:
                # Already a dict
                self.depth_indexes = depth_indexes_df

        # FIX 2: Filter proteins without labels
        all_pdb_ids = self.protein_df['pdb_id'].tolist()
        valid_pdb_ids = [pid for pid in all_pdb_ids if pid in self.depth_indexes]
        excluded_count = len(all_pdb_ids) - len(valid_pdb_ids)

        print(f"Filtering proteins without labels:")
        print(f"  Total proteins: {len(all_pdb_ids)}")
        print(f"  Proteins with labels: {len(valid_pdb_ids)}")
        print(f"  Proteins excluded: {excluded_count}")

        # Filter dataframe to only valid proteins
        self.protein_df = self.protein_df[self.protein_df['pdb_id'].isin(valid_pdb_ids)]

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

        # Setup feature normalization
        self.normalizer = None
        if normalize_features:
            if normalizer_path is None:
                normalizer_path = os.path.join(root, f'normalizer_{split}.pkl')

            self.normalizer_path = normalizer_path

            if split == 'train':
                # Training: fit normalizer on training data
                print(f"Fitting normalizer on training data...")
                self.normalizer = self._fit_normalizer()
                self.normalizer.save(self.normalizer_path)
                print(f"  Saved normalizer to {self.normalizer_path}")
            else:
                # Val/Test: load normalizer from training
                train_normalizer_path = os.path.join(root, 'normalizer_train.pkl')
                if os.path.exists(train_normalizer_path):
                    self.normalizer = FeatureNormalizer()
                    self.normalizer.load(train_normalizer_path)
                    print(f"Loaded normalizer from {train_normalizer_path}")
                else:
                    print(f"WARNING: No normalizer found at {train_normalizer_path}")
                    print(f"         Features will not be normalized!")

        # Print feature dimensions
        dims = get_feature_dimensions()
        print(f"\nFeature dimensions:")
        print(f"  Numerical: {dims['numerical']}")
        print(f"  Categorical: {dims['atom_types'] + dims['elements'] + dims['residues']}")
        print(f"  Geometric: {dims['geometric']}")
        print(f"  Total: {dims['total']}")

        super().__init__(root, transform, pre_transform, pre_filter)

    def _fit_normalizer(self) -> FeatureNormalizer:
        """
        Fit normalizer on a subset of training data.
        Uses first 100 proteins to compute statistics.
        """
        try:
            from .feature_engineering import SELECTED_NUMERICAL_FEATURES
        except ImportError:
            from feature_engineering import SELECTED_NUMERICAL_FEATURES

        normalizer = FeatureNormalizer()
        all_features = []

        # Sample proteins for fitting (use first 100)
        sample_ids = self.protein_ids[:min(100, len(self.protein_ids))]

        for pdb_id in sample_ids:
            try:
                protein_dir = os.path.join(self.root, 'dataset', 'sadic_data', pdb_id)
                nodes_path = os.path.join(protein_dir, f'{pdb_id}__graphein__ATOM_nodes.csv')
                nodes_df = pd.read_csv(nodes_path, index_col=0)

                # Extract numerical features
                numerical_features = nodes_df[SELECTED_NUMERICAL_FEATURES].values
                all_features.append(numerical_features)
            except Exception as e:
                print(f"  Warning: Skipping {pdb_id} for normalization: {e}")
                continue

        if len(all_features) > 0:
            all_features = np.concatenate(all_features, axis=0)
            normalizer.fit(all_features)
        else:
            raise RuntimeError("Could not fit normalizer - no valid proteins found")

        return normalizer

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
            data = torch.load(processed_path, weights_only=False)
        else:
            data = self._load_protein_graph(pdb_id)

        return data

    def _load_protein_graph(self, pdb_id: str) -> Data:
        """
        Load a protein graph from CSV files with all fixes applied.

        Args:
            pdb_id (str): Protein PDB ID

        Returns:
            Data: PyTorch Geometric Data object with engineered features
        """
        protein_dir = os.path.join(self.root, 'dataset', 'sadic_data', pdb_id)

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

        # Create node ID to index mapping
        node_ids = nodes_df['original_index'].values
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Create edge index and extract distances
        edge_index = []
        edge_distances = []

        for _, edge in edges_df.iterrows():
            src = node_to_idx.get(edge['idx_0'])
            dst = node_to_idx.get(edge['idx_1'])

            if src is not None and dst is not None:
                edge_index.append([src, dst])
                distance = edge['distance'] if not pd.isna(edge['distance']) else 0.0
                edge_distances.append(distance)

        edge_index_np = np.array(edge_index).T if len(edge_index) > 0 else np.zeros((2, 0), dtype=np.int64)
        edge_distances_np = np.array(edge_distances) if len(edge_distances) > 0 else np.zeros(0)

        # FIX 3: Extract all features using feature engineering pipeline
        features, feature_names = extract_all_features(
            nodes_df=nodes_df,
            edge_index=edge_index_np,
            edge_distances=edge_distances_np,
            normalizer=self.normalizer,
            normalize=self.normalize_features
        )

        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        edge_attr = torch.tensor(edge_distances_np, dtype=torch.float).unsqueeze(-1)

        # FIX 4: Proper target matching
        if pdb_id in self.depth_indexes:
            depth_dict = self.depth_indexes[pdb_id]
            y = []

            for atom_name in node_ids:
                if atom_name in depth_dict:
                    y.append(depth_dict[atom_name])
                else:
                    # This should rarely happen after filtering
                    print(f"Warning: {atom_name} not in depth_indexes for {pdb_id}")
                    y.append(0.0)  # Fallback

            y = torch.tensor(y, dtype=torch.float)
        else:
            # Should never happen after filtering valid_pdb_ids
            raise ValueError(f"PDB {pdb_id} has no depth data - this should not happen!")

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
    # Test dataset loading with fixes
    print("="*80)
    print("Testing FIXED Dataset Implementation")
    print("="*80)

    dataset = ProteinAtomDataset(root='../../dataset/', split='train')
    print(f"\nDataset size: {len(dataset)}")

    print("\nLoading sample...")
    sample = dataset[0]
    print(f"\nSample graph:")
    print(f"  PDB ID: {sample.pdb_id}")
    print(f"  Number of atoms: {sample.num_nodes}")
    print(f"  Number of edges: {sample.num_edges}")
    print(f"  Node features shape: {sample.x.shape}")
    print(f"  Edge features shape: {sample.edge_attr.shape}")
    print(f"  Target shape: {sample.y.shape}")
    print(f"  Target range: [{sample.y.min():.4f}, {sample.y.max():.4f}]")
    print(f"  Target mean: {sample.y.mean():.4f}")

    print("\n" + "="*80)
    print("SUCCESS: Dataset loading works with all fixes applied!")
    print("="*80)
