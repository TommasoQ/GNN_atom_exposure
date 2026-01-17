"""
Data transforms for preprocessing protein graph features with feature engineering.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path


class EdgeTypeEncoder:
    """
    Encodes edge types as multi-hot vectors for bond properties.
    """

    # Define bond type categories
    BOND_TYPES = ['covalent', 'peptide_bond', 'ionic', 'hbond', 'hydrophobic', 'aromatic', 'RING']

    def __init__(self):
        self.fitted = True  # Always ready since we use fixed categories

    def transform(self, edge_types: List[str]) -> np.ndarray:
        """
        Transform edge types to multi-hot vectors.

        Args:
            edge_types: List of edge type strings (e.g., "covalent, RING")

        Returns:
            Multi-hot array of shape (num_edges, num_bond_types)
        """
        multi_hot = np.zeros((len(edge_types), len(self.BOND_TYPES)), dtype=np.float32)

        for i, edge_type in enumerate(edge_types):
            for j, bond_type in enumerate(self.BOND_TYPES):
                if bond_type.lower() in edge_type.lower():
                    multi_hot[i, j] = 1.0

        return multi_hot

    @property
    def num_types(self) -> int:
        """Number of bond type categories."""
        return len(self.BOND_TYPES)

    def save(self, path: Path) -> None:
        """Save encoder to file."""
        with open(path, 'wb') as f:
            pickle.dump({'bond_types': self.BOND_TYPES}, f)

    def load(self, path: Path) -> 'EdgeTypeEncoder':
        """Load encoder from file."""
        # No state to load, always use fixed categories
        return self


class FeatureNormalizer:
    """
    Feature engineering and normalization for protein graphs.

    Key improvements:
    - Aggregates correlated hydrophobicity scales
    - Converts absolute coordinates to relative (centroid-based)
    - Adds local geometric features
    - Removes redundant features
    """

    # Categorical columns (excluded from numerical processing)
    CATEGORICAL_COLS = ['original_index', 'chain_id', 'residue_name', 'atom_type', 'element_symbol', 'Unnamed: 0']

    # Hydrophobicity columns to aggregate (highly correlated)
    HYDROPHOBICITY_COLS = [
        'expasy:hphob_eisenberg', 'expasy:hphob_sweet', 'expasy:hphob_woods',
        'expasy:hphob_doolittle', 'expasy:hphob_leo', 'expasy:hphob_fauchere',
        'expasy:hphob_guy', 'expasy:hphob_janin', 'expasy:hphob_roseman',
        'expasy:hphob_tanford', 'expasy:hphob_parker', 'expasy:hphob_chothia',
        'expasy:hphob_rose'
    ]

    # Secondary structure propensity columns to aggregate
    HELIX_COLS = ['expasy:alpha_helixfasman', 'expasy:alpha_helixroux', 'expasy:alpha_helixlevitt']
    SHEET_COLS = ['expasy:beta_sheetfasman', 'expasy:beta_sheetroux', 'expasy:beta_sheetlevitt']
    TURN_COLS = ['expasy:beta_turnfasman', 'expasy:beta_turnroux', 'expasy:beta_turnlevitt']

    # Core features to keep (non-redundant)
    CORE_FEATURES = [
        'residue_number', 'b_factor',
        'hbond_donors', 'hbond_acceptors',
        'meiler:dim_1', 'meiler:dim_2', 'meiler:dim_3', 'meiler:dim_4',
        'meiler:dim_5', 'meiler:dim_6', 'meiler:dim_7',
        'expasy:molecularweight', 'expasy:bulkiness',
        'expasy:polarityzimmerman', 'expasy:refractivity',
        'expasy:isoelectric_points', 'expasy:averageflexibility',
        'expasy:buriedresidues', 'expasy:accessibleresidues'
    ]

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.element_encoder = LabelEncoder()
        self.residue_encoder = LabelEncoder()
        self.fitted = False

        self.feature_names: List[str] = []
        self.element_classes: List[str] = []
        self.residue_classes: List[str] = []

    def _engineer_node_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to node dataframe.

        Returns new dataframe with engineered features.
        """
        features = {}

        # 1. Core features (keep as-is)
        for col in self.CORE_FEATURES:
            if col in df.columns:
                features[col] = df[col].values

        # 2. Aggregate hydrophobicity (mean of available scales)
        hydro_cols = [c for c in self.HYDROPHOBICITY_COLS if c in df.columns]
        if hydro_cols:
            features['hydrophobicity_mean'] = df[hydro_cols].mean(axis=1).values
            features['hydrophobicity_std'] = df[hydro_cols].std(axis=1).values

        # 3. Aggregate secondary structure propensities
        helix_cols = [c for c in self.HELIX_COLS if c in df.columns]
        sheet_cols = [c for c in self.SHEET_COLS if c in df.columns]
        turn_cols = [c for c in self.TURN_COLS if c in df.columns]

        if helix_cols:
            features['helix_propensity'] = df[helix_cols].mean(axis=1).values
        if sheet_cols:
            features['sheet_propensity'] = df[sheet_cols].mean(axis=1).values
        if turn_cols:
            features['turn_propensity'] = df[turn_cols].mean(axis=1).values

        # 4. Geometric features from coordinates
        coords = df[['x_coord', 'y_coord', 'z_coord']].values
        centroid = coords.mean(axis=0)

        # Distance from centroid (proxy for burial)
        dist_from_centroid = np.linalg.norm(coords - centroid, axis=1)
        features['dist_from_centroid'] = dist_from_centroid

        # Relative coordinates (normalized by protein size)
        rel_coords = coords - centroid
        protein_radius = dist_from_centroid.max() + 1e-6
        features['rel_x'] = rel_coords[:, 0] / protein_radius
        features['rel_y'] = rel_coords[:, 1] / protein_radius
        features['rel_z'] = rel_coords[:, 2] / protein_radius

        # Spherical coordinates (normalized)
        r = dist_from_centroid / protein_radius
        theta = np.arctan2(rel_coords[:, 1], rel_coords[:, 0]) / np.pi  # [-1, 1]
        phi = np.arccos(np.clip(rel_coords[:, 2] / (dist_from_centroid + 1e-6), -1, 1)) / np.pi  # [0, 1]
        features['spherical_r'] = r
        features['spherical_theta'] = theta
        features['spherical_phi'] = phi

        return pd.DataFrame(features)

    def fit_nodes(self, node_dfs: List[pd.DataFrame]) -> 'FeatureNormalizer':
        """
        Fit normalizer on list of node dataframes.
        """
        # Engineer features for all proteins
        engineered_dfs = [self._engineer_node_features(df) for df in node_dfs]

        # Get feature names
        self.feature_names = list(engineered_dfs[0].columns)

        # Concatenate all numerical features
        all_features = np.vstack([df.values for df in engineered_dfs])

        # Handle NaN
        all_features = np.nan_to_num(all_features, nan=0.0)

        # Fit scaler
        self.node_scaler.fit(all_features)

        # Fit element encoder
        all_elements = []
        for df in node_dfs:
            all_elements.extend(df['element_symbol'].tolist())
        self.element_encoder.fit(sorted(set(all_elements)))
        self.element_classes = list(self.element_encoder.classes_)

        # Fit residue encoder
        all_residues = []
        for df in node_dfs:
            all_residues.extend(df['residue_name'].tolist())
        self.residue_encoder.fit(sorted(set(all_residues)))
        self.residue_classes = list(self.residue_encoder.classes_)

        self.fitted = True
        return self

    def transform_nodes(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform node dataframe to tensors.

        Returns:
            Tuple of (numerical_features, element_indices, residue_indices)
        """
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer not fitted. Call fit_nodes() first.")

        # Engineer features
        engineered_df = self._engineer_node_features(df)

        # Normalize numerical features
        features = engineered_df.values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features_normalized = self.node_scaler.transform(features)

        # Encode elements
        elements = df['element_symbol'].values
        element_indices = self.element_encoder.transform(elements)

        # Encode residues
        residues = df['residue_name'].values
        residue_indices = self.residue_encoder.transform(residues)

        return (
            torch.tensor(features_normalized, dtype=torch.float32),
            torch.tensor(element_indices, dtype=torch.long),
            torch.tensor(residue_indices, dtype=torch.long)
        )

    def fit_edges(self, edge_dfs: List[pd.DataFrame]) -> 'FeatureNormalizer':
        """Fit scaler for edge numerical features."""
        all_edge_features = []

        for df in edge_dfs:
            distance = df['distance'].values.astype(np.float32)
            bond_length = df['bond_length'].fillna(0).values.astype(np.float32)

            # Log-transform distance for better distribution
            log_distance = np.log1p(distance)

            # Has bond length flag
            has_bond = (bond_length > 0).astype(np.float32)

            edge_features = np.stack([distance, log_distance, bond_length, has_bond], axis=1)
            all_edge_features.append(edge_features)

        combined = np.vstack(all_edge_features)
        self.edge_scaler.fit(combined)
        return self

    def transform_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Transform edge dataframe to tensor with engineered features.

        Returns:
            Tensor of shape (num_edges, 4) with [distance, log_distance, bond_length, has_bond]
        """
        distance = df['distance'].values.astype(np.float32)
        bond_length = df['bond_length'].fillna(0).values.astype(np.float32)

        log_distance = np.log1p(distance)
        has_bond = (bond_length > 0).astype(np.float32)

        edge_features = np.stack([distance, log_distance, bond_length, has_bond], axis=1)
        edge_features_normalized = self.edge_scaler.transform(edge_features)
        edge_features_normalized = np.nan_to_num(edge_features_normalized, nan=0.0)

        return torch.tensor(edge_features_normalized, dtype=torch.float32)

    @property
    def num_node_features(self) -> int:
        """Number of numerical node features after engineering."""
        return len(self.feature_names)

    @property
    def num_elements(self) -> int:
        """Number of unique element types."""
        return len(self.element_classes)

    @property
    def num_residues(self) -> int:
        """Number of unique residue types."""
        return len(self.residue_classes)

    def save(self, path: Path) -> None:
        """Save normalizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'node_scaler': self.node_scaler,
                'edge_scaler': self.edge_scaler,
                'element_encoder': self.element_encoder,
                'residue_encoder': self.residue_encoder,
                'feature_names': self.feature_names,
                'element_classes': self.element_classes,
                'residue_classes': self.residue_classes,
                'fitted': self.fitted
            }, f)

    def load(self, path: Path) -> 'FeatureNormalizer':
        """Load normalizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.node_scaler = data['node_scaler']
            self.edge_scaler = data['edge_scaler']
            self.element_encoder = data['element_encoder']
            self.residue_encoder = data.get('residue_encoder', LabelEncoder())
            self.feature_names = data['feature_names']
            self.element_classes = data['element_classes']
            self.residue_classes = data.get('residue_classes', [])
            self.fitted = data['fitted']
        return self
