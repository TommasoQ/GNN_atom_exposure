"""
PyTorch Geometric Dataset for Protein Atom Exposure Prediction
FIXED VERSION - Phase 2 + Enhanced Edge Features + Optimized
"""
import os
import pickle
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Bond types for edge feature extraction (7 types, one-hot encoded)
BOND_TYPES = ['covalent', 'peptide_bond', 'hydrophobic', 'aromatic', 'hbond', 'ionic', 'ring']


def extract_edge_features(edges_df: pd.DataFrame) -> np.ndarray:
    """
    Extract 12 edge features from edge DataFrame (VECTORIZED).

    Features (12 total):
        - 7 bond type one-hot features: covalent, peptide_bond, hydrophobic, aromatic, hbond, ionic, ring
        - 4 numerical features: distance, bond_length, normalized_distance, relative_distance
        - 1 radius graph feature: in_radius (1 if distance < 8.0Å, 0 otherwise)

    Args:
        edges_df: DataFrame with 'kind', 'distance', 'bond_length' columns

    Returns:
        np.ndarray of shape (num_edges, 12)
    """
    n_edges = len(edges_df)
    edge_features = np.zeros((n_edges, 12), dtype=np.float32)

    # Vectorized: Normalize 'kind' column to lowercase strings
    kind_series = edges_df['kind'].fillna('').astype(str).str.lower()

    # Vectorized: 7 bond type one-hot features (indices 0-6)
    for j, bond_type in enumerate(BOND_TYPES):
        edge_features[:, j] = kind_series.str.contains(bond_type, regex=False).astype(np.float32)

    # Vectorized: Extract numerical columns with NaN handling
    distance = edges_df['distance'].fillna(0.0).values.astype(np.float32)
    bond_length = edges_df['bond_length'].fillna(0.0).values.astype(np.float32)

    # Vectorized: 4 numerical features (indices 7-10)
    edge_features[:, 7] = distance                    # Raw distance
    edge_features[:, 8] = bond_length                 # Bond length (0 if not covalent)
    edge_features[:, 9] = distance / 10.0             # Normalized distance
    edge_features[:, 10] = distance - 3.8             # Relative to avg Cα-Cα

    # Vectorized: Radius graph feature (index 11)
    edge_features[:, 11] = (distance < 8.0).astype(np.float32)

    return edge_features


# Import feature engineering modules
try:
    from .feature_engineering import (
        extract_all_features,
        FeatureNormalizer,
        get_feature_dimensions,
        SELECTED_NUMERICAL_FEATURES,
        REDUCED_NUMERICAL_FEATURES
    )
    from .aggregated_transforms import (
        AggregatedFeatureNormalizer,
        get_aggregated_feature_dimensions
    )
except ImportError:
    # For direct execution
    from feature_engineering import (
        extract_all_features,
        FeatureNormalizer,
        get_feature_dimensions,
        SELECTED_NUMERICAL_FEATURES,
        REDUCED_NUMERICAL_FEATURES
    )
    from aggregated_transforms import (
        AggregatedFeatureNormalizer,
        get_aggregated_feature_dimensions
    )


def _process_single_protein(args: tuple) -> str:
    """
    Helper function for parallel protein processing.
    Must be at module level for multiprocessing to pickle it.

    Args:
        args: Tuple of (pdb_id, config_dict) where config_dict contains all needed parameters

    Returns:
        pdb_id if successful, None otherwise
    """
    pdb_id, config = args

    try:
        root = config['root']
        processed_dir = config['processed_dir']
        depth_indexes = config['depth_indexes']
        normalizer = config['normalizer']
        normalize_features = config['normalize_features']
        use_reduced_features = config['use_reduced_features']
        include_atom_type = config['include_atom_type']
        include_geometric = config['include_geometric']
        include_backbone_angles = config['include_backbone_angles']
        use_minimal_features = config.get('use_minimal_features', False)

        protein_dir = os.path.join(root, 'sadic_data', pdb_id)

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

        # Vectorized: Create edge index and filter valid edges
        src_ids = edges_df['idx_0'].values
        dst_ids = edges_df['idx_1'].values

        map_func = np.vectorize(lambda x: node_to_idx.get(x, -1))
        src_indices = map_func(src_ids)
        dst_indices = map_func(dst_ids)

        valid_mask = (src_indices >= 0) & (dst_indices >= 0)
        valid_edge_indices = np.where(valid_mask)[0]

        if len(valid_edge_indices) > 0:
            edge_index_np = np.stack([src_indices[valid_mask], dst_indices[valid_mask]], axis=0)
        else:
            edge_index_np = np.zeros((2, 0), dtype=np.int64)

        # Extract edge features
        if len(valid_edge_indices) > 0:
            valid_edges_df = edges_df.iloc[valid_edge_indices]
            edge_features = extract_edge_features(valid_edges_df)
        else:
            edge_features = np.zeros((0, 12), dtype=np.float32)

        edge_distances_np = edge_features[:, 7] if len(edge_features) > 0 else np.zeros(0)

        # Extract node features
        features, feature_names = extract_all_features(
            nodes_df=nodes_df,
            edge_index=edge_index_np,
            edge_distances=edge_distances_np,
            normalizer=normalizer,
            normalize=normalize_features,
            use_reduced_features=use_reduced_features,
            include_atom_type=include_atom_type,
            include_geometric=include_geometric,
            include_backbone_angles=include_backbone_angles,
            use_minimal_features=use_minimal_features
        )
        x = torch.tensor(features, dtype=torch.float)

        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # Skip edge features in minimal mode (they have 0 importance)
        if use_minimal_features:
            edge_attr = torch.zeros((edge_index_np.shape[1], 0), dtype=torch.float)
        else:
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Extract targets
        depth_dict = depth_indexes[pdb_id]
        y = []
        for atom_name in node_ids:
            if atom_name not in depth_dict:
                return None  # Skip problematic proteins
            y.append(depth_dict[atom_name])
        y = torch.tensor(y, dtype=torch.float)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pdb_id=pdb_id,
            num_nodes=len(nodes_df)
        )

        # Save to disk
        torch.save(data, os.path.join(processed_dir, f'{pdb_id}.pt'))
        return pdb_id

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None


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
        feature_config (dict): Feature configuration with keys:
            - use_reduced_features (bool): Use redundancy-reduced features
            - include_atom_type (bool): Include atom type one-hot encoding
            - include_geometric (bool): Include geometric features
            - use_aggregated (bool): Use aggregated feature transforms (31 numerical + embeddings)
        transform (callable, optional): Transform to apply to each graph
        pre_transform (callable, optional): Transform to apply before saving
        pre_filter (callable, optional): Filter to apply before saving
        verbose (bool): Whether to print detailed loading info (default: True for train, False otherwise)
    """

    # Class-level cache to avoid reloading depth_indexes for each split
    _depth_indexes_cache = None
    _dataset_info_printed = False

    def __init__(
        self,
        root: str,
        split: str = 'train',
        normalize_features: bool = True,
        normalizer_path: Optional[str] = None,
        feature_config: Optional[dict] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.split = split
        self.root = root
        self.normalize_features = normalize_features

        # Feature configuration (defaults to full feature set)
        self.feature_config = feature_config or {}
        self.use_reduced_features = self.feature_config.get('use_reduced_features', False)
        self.include_atom_type = self.feature_config.get('include_atom_type', True)
        self.include_geometric = self.feature_config.get('include_geometric', True)
        self.use_aggregated = self.feature_config.get('use_aggregated', False)
        self.include_backbone_angles = self.feature_config.get('include_backbone_angles', False)
        self.use_minimal_features = self.feature_config.get('use_minimal_features', False)

        # Load protein list
        protein_csv_path = os.path.join(root, 'protein_sample_5000.csv')
        self.protein_df = pd.read_csv(protein_csv_path)

        # FIX 1: Load depth_indexes (use class-level cache to avoid reloading)
        if ProteinAtomDataset._depth_indexes_cache is None:
            depth_dict_path = os.path.join(root, 'depth_indexes_dict.pkl')
            depth_df_path = os.path.join(root, 'depth_indexes.pkl')

            if os.path.exists(depth_dict_path):
                with open(depth_dict_path, 'rb') as f:
                    ProteinAtomDataset._depth_indexes_cache = pickle.load(f)
            else:
                # Fallback: load DataFrame and convert (slow)
                print(f"WARNING: depth_indexes_dict.pkl not found, converting from DataFrame...")
                with open(depth_df_path, 'rb') as f:
                    depth_indexes_df = pickle.load(f)

                if isinstance(depth_indexes_df, pd.DataFrame):
                    from tqdm import tqdm
                    ProteinAtomDataset._depth_indexes_cache = {}
                    unique_pdb_ids = depth_indexes_df['pdb_id'].unique()
                    for pdb_id in tqdm(unique_pdb_ids, desc="Converting depth_indexes", unit="protein"):
                        pdb_data = depth_indexes_df[depth_indexes_df['pdb_id'] == pdb_id]
                        ProteinAtomDataset._depth_indexes_cache[pdb_id] = dict(zip(
                            pdb_data['atom_name'],
                            pdb_data['depth_index']
                        ))
                else:
                    ProteinAtomDataset._depth_indexes_cache = depth_indexes_df

                # Save the converted dict for faster loading next time
                print(f"Saving converted dict to {depth_dict_path}...")
                with open(depth_dict_path, 'wb') as f:
                    pickle.dump(ProteinAtomDataset._depth_indexes_cache, f)
                print(f"Saved! Next run will load directly from dict.")

        self.depth_indexes = ProteinAtomDataset._depth_indexes_cache

        # FIX 2: Filter proteins without labels
        all_pdb_ids = self.protein_df['pdb_id'].tolist()
        valid_pdb_ids = [pid for pid in all_pdb_ids if pid in self.depth_indexes]

        # FIX 5: Exclude HETATM-dominant proteins (very few protein atoms, mostly ligands)
        # These 38 proteins have raw_atoms/graphein_atoms ratio > 3.8
        # They're valid but contribute minimal training signal (4-227 atoms)
        HETATM_DOMINANT_PROTEINS = {
            '3mbs', '6phm', '6phq', '2kql', '6phn', '3try', '4ttk', '2q33',
            '1a7z', '5m2h', '1hzs', '1bfw', '1hhy', '5m2k', '1qd8', '1cya',
            '1rru', '7c4u', '7c4v', '1ghg', '1hhz', '6mw0', '1cw8', '1cvq',
            '1al4', '6ug2', '4g14', '2l2w', '6udz', '1alz', '6ud9', '6ufu',
            '1m24', '1kyj', '4k7t', '7rms', '7rmr', '7l98'
        }
        valid_pdb_ids = [pid for pid in valid_pdb_ids if pid not in HETATM_DOMINANT_PROTEINS]

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
        self.aggregated_normalizer = None

        # Determine normalizer suffix based on feature mode
        normalizer_suffix = '_aggregated' if self.use_aggregated else ''

        # Minimal mode doesn't need normalization (only geometric features)
        if self.use_minimal_features:
            normalize_features = False
            self.normalize_features = False

        if normalize_features:
            if normalizer_path is None:
                normalizer_path = os.path.join(root, f'normalizer_{split}{normalizer_suffix}.pkl')

            self.normalizer_path = normalizer_path

            if split == 'train':
                # Training: fit normalizer on training data
                if self.use_aggregated:
                    self.aggregated_normalizer = self._fit_aggregated_normalizer()
                    self.aggregated_normalizer.save(self.normalizer_path)
                else:
                    self.normalizer = self._fit_normalizer()
                    self.normalizer.save(self.normalizer_path)
            else:
                # Val/Test: load normalizer from training
                train_normalizer_path = os.path.join(root, f'normalizer_train{normalizer_suffix}.pkl')
                if os.path.exists(train_normalizer_path):
                    if self.use_aggregated:
                        self.aggregated_normalizer = AggregatedFeatureNormalizer()
                        self.aggregated_normalizer.load(train_normalizer_path)
                    else:
                        self.normalizer = FeatureNormalizer()
                        self.normalizer.load(train_normalizer_path)
                else:
                    print(f"WARNING: No normalizer found at {train_normalizer_path}")

        # Print dataset info only once (on first split loaded)
        if not ProteinAtomDataset._dataset_info_printed:
            if self.use_aggregated:
                dims = get_aggregated_feature_dimensions()
                config_str = "aggregated=True (31 numerical + embeddings)"
            elif self.use_minimal_features:
                dims = get_feature_dimensions(use_minimal_features=True)
                config_str = "minimal=True (5 geometric features only)"
            else:
                dims = get_feature_dimensions(
                    use_reduced_features=self.use_reduced_features,
                    include_atom_type=self.include_atom_type,
                    include_geometric=self.include_geometric,
                    include_backbone_angles=self.include_backbone_angles
                )
                config_str = f"reduced={self.use_reduced_features}, atom_type={self.include_atom_type}, geometric={self.include_geometric}, backbone={self.include_backbone_angles}"
            print(f"Dataset: {len(valid_pdb_ids)} proteins, {dims['total']} features ({config_str})")
            ProteinAtomDataset._dataset_info_printed = True

        super().__init__(root, transform, pre_transform, pre_filter)

    def _fit_normalizer(self) -> FeatureNormalizer:
        """
        Fit normalizer on a subset of training data.
        Uses first 100 proteins to compute statistics.
        """
        # Select feature list based on configuration
        numerical_feature_list = REDUCED_NUMERICAL_FEATURES if self.use_reduced_features else SELECTED_NUMERICAL_FEATURES

        normalizer = FeatureNormalizer()
        all_features = []

        # Sample proteins for fitting (use first 100)
        sample_ids = self.protein_ids[:min(100, len(self.protein_ids))]

        for pdb_id in sample_ids:
            try:
                protein_dir = os.path.join(self.root, 'sadic_data', pdb_id)
                nodes_path = os.path.join(protein_dir, f'{pdb_id}__graphein__ATOM_nodes.csv')
                nodes_df = pd.read_csv(nodes_path, index_col=0)

                # Extract numerical features based on configuration
                numerical_features = nodes_df[numerical_feature_list].values
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

    def _fit_aggregated_normalizer(self) -> AggregatedFeatureNormalizer:
        """
        Fit aggregated normalizer on a subset of training data.
        Uses first 100 proteins to compute statistics.
        """
        normalizer = AggregatedFeatureNormalizer()
        node_dfs = []

        # Sample proteins for fitting (use first 100)
        sample_ids = self.protein_ids[:min(100, len(self.protein_ids))]

        for pdb_id in sample_ids:
            try:
                protein_dir = os.path.join(self.root, 'sadic_data', pdb_id)
                nodes_path = os.path.join(protein_dir, f'{pdb_id}__graphein__ATOM_nodes.csv')
                nodes_df = pd.read_csv(nodes_path, index_col=0)
                node_dfs.append(nodes_df)
            except Exception as e:
                print(f"  Warning: Skipping {pdb_id} for normalization: {e}")
                continue

        if len(node_dfs) > 0:
            normalizer.fit_nodes(node_dfs)
        else:
            raise RuntimeError("Could not fit aggregated normalizer - no valid proteins found")

        return normalizer

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names."""
        return ['protein_sample_5000.csv', 'depth_indexes.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        """
        List of processed file names.

        NOTE: Returns ALL protein files (not split-specific) to enable proper caching.
        The dataset split (train/val/test) is handled by indexing, not by processing
        different files.
        """
        # Get ALL valid protein IDs (not just this split's)
        all_pdb_ids = self.protein_df['pdb_id'].tolist()
        valid_pdb_ids = [pid for pid in all_pdb_ids if pid in self.depth_indexes]
        return [f'{pdb_id}.pt' for pdb_id in valid_pdb_ids]

    def download(self):
        """Download dataset (not needed as data is already present)."""
        pass

    def process(self, parallel: bool = True, max_workers: int = None):
        """
        Process raw data into PyTorch Geometric Data objects.

        Processes ALL valid proteins (not just current split) to enable caching.
        Each split accesses its subset via indexing.

        Args:
            parallel: Use parallel processing (default: True)
            max_workers: Number of worker processes (default: CPU count - 1)
        """
        from tqdm import tqdm

        # Get ALL valid protein IDs
        all_pdb_ids = self.protein_df['pdb_id'].tolist()
        valid_pdb_ids = [pid for pid in all_pdb_ids if pid in self.depth_indexes]

        print(f"Processing {len(valid_pdb_ids)} proteins (shared across all splits)...")

        # Prepare config dict for parallel processing (only non-aggregated mode)
        if parallel and not self.use_aggregated:
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)

            config = {
                'root': self.root,
                'processed_dir': self.processed_dir,
                'depth_indexes': self.depth_indexes,
                'normalizer': self.normalizer,
                'normalize_features': self.normalize_features,
                'use_reduced_features': self.use_reduced_features,
                'include_atom_type': self.include_atom_type,
                'include_geometric': self.include_geometric,
                'include_backbone_angles': self.include_backbone_angles,
                'use_minimal_features': self.use_minimal_features
            }

            # Create argument tuples for parallel processing
            args_list = [(pdb_id, config) for pdb_id in valid_pdb_ids]

            print(f"Using {max_workers} parallel workers...")
            successful = 0
            failed = 0

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(_process_single_protein, args): args[0]
                          for args in args_list}

                # Process results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures),
                                  desc="Processing proteins", unit="protein"):
                    pdb_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"Error processing {pdb_id}: {e}")
                        failed += 1

            print(f"Completed: {successful} successful, {failed} failed")

        else:
            # Sequential processing (fallback or for aggregated mode)
            if self.use_aggregated:
                print("Using sequential processing (aggregated mode)...")
            else:
                print("Using sequential processing...")

            for pdb_id in tqdm(valid_pdb_ids, desc="Processing proteins", unit="protein"):
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

        # If minimal mode but cached data has full features, select the 5 minimal ones
        if self.use_minimal_features and data.x.shape[1] != 5:
            from src.data.feature_engineering import MINIMAL_GEOMETRIC_INDICES
            # Geometric block starts after: numerical(24) + atom_types(31) + elements(5) + residues(21) = 81
            geom_offset = data.x.shape[1] - 8 - 4  # 8 geometric + 4 backbone at the end
            if data.x.shape[1] == 93:
                geom_offset = 81  # 24 + 31 + 5 + 21
            abs_indices = [geom_offset + i for i in MINIMAL_GEOMETRIC_INDICES]
            data.x = data.x[:, abs_indices]
            # Zero out edge features (they have 0 importance in minimal mode)
            data.edge_attr = torch.zeros((data.edge_index.shape[1], 0), dtype=torch.float)

        return data

    def _load_protein_graph(self, pdb_id: str) -> Data:
        """
        Load a protein graph from CSV files with all fixes applied.

        Args:
            pdb_id (str): Protein PDB ID

        Returns:
            Data: PyTorch Geometric Data object with engineered features
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

        # Create node ID to index mapping (vectorized)
        node_ids = nodes_df['original_index'].values
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Vectorized: Create edge index and filter valid edges
        src_ids = edges_df['idx_0'].values
        dst_ids = edges_df['idx_1'].values

        # Map node IDs to indices (vectorized with np.vectorize)
        map_func = np.vectorize(lambda x: node_to_idx.get(x, -1))
        src_indices = map_func(src_ids)
        dst_indices = map_func(dst_ids)

        # Filter valid edges (both src and dst exist in node mapping)
        valid_mask = (src_indices >= 0) & (dst_indices >= 0)
        valid_edge_indices = np.where(valid_mask)[0]

        if len(valid_edge_indices) > 0:
            edge_index_np = np.stack([src_indices[valid_mask], dst_indices[valid_mask]], axis=0)
        else:
            edge_index_np = np.zeros((2, 0), dtype=np.int64)
        
        # Extract 12-dimensional edge features (7 bond types + 4 numerical + 1 radius graph)
        if len(valid_edge_indices) > 0:
            valid_edges_df = edges_df.iloc[valid_edge_indices]
            edge_features = extract_edge_features(valid_edges_df)
        else:
            edge_features = np.zeros((0, 12), dtype=np.float32)
        
        # Also extract distances for node feature computation (geometric features)
        edge_distances_np = edge_features[:, 7] if len(edge_features) > 0 else np.zeros(0)

        # Extract features based on mode
        element_idx = None
        residue_idx = None

        if self.use_aggregated and self.aggregated_normalizer is not None:
            # Use aggregated feature transforms (31 numerical + embedding indices)
            numerical_features, element_idx, residue_idx = self.aggregated_normalizer.transform_nodes(nodes_df)
            x = numerical_features  # Already a tensor
        else:
            # FIX 3: Extract all features using feature engineering pipeline
            features, feature_names = extract_all_features(
                nodes_df=nodes_df,
                edge_index=edge_index_np,
                edge_distances=edge_distances_np,
                normalizer=self.normalizer,
                normalize=self.normalize_features,
                use_reduced_features=self.use_reduced_features,
                include_atom_type=self.include_atom_type,
                include_geometric=self.include_geometric,
                include_backbone_angles=self.include_backbone_angles,
                use_minimal_features=self.use_minimal_features
            )
            x = torch.tensor(features, dtype=torch.float)

        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # Skip edge features in minimal mode (they have 0 importance)
        if self.use_minimal_features:
            edge_attr = torch.zeros((edge_index_np.shape[1], 0), dtype=torch.float)
        else:
            edge_attr = torch.tensor(edge_features, dtype=torch.float)  # Now 12-dim

        # FIX 4: Proper target matching
        if pdb_id in self.depth_indexes:
            depth_dict = self.depth_indexes[pdb_id]
            y = []

            for atom_name in node_ids:
                if atom_name not in depth_dict:
                    raise ValueError(
                        f"Atom {atom_name} in protein {pdb_id} has no depth label. "
                        f"This should not happen after filtering. Check data integrity."
                    )
                y.append(depth_dict[atom_name])

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

        # Add embedding indices if using aggregated transforms
        if element_idx is not None:
            data.element_idx = element_idx
        if residue_idx is not None:
            data.residue_idx = residue_idx

        return data


if __name__ == '__main__':
    # Test dataset loading with fixes
    print("="*80)
    print("Testing FIXED Dataset Implementation")
    print("="*80)

    dataset = ProteinAtomDataset(root='../../', split='train')
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
