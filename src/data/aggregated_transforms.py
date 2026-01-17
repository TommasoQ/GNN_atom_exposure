"""
Aggregated Feature Transforms for Protein Graphs

This module implements feature engineering with aggregation of correlated features:
- Hydrophobicity: 13 scales aggregated to mean + std (2 features)
- Secondary structure: helix/sheet/turn propensities aggregated (3 features)
- Geometric: centroid-relative coordinates + spherical coords (7 features)
- Categorical: LabelEncoder indices for embedding layers (not one-hot)

Total: 31 numerical features + element/residue indices for embeddings
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path


class AggregatedFeatureNormalizer:
    """
    Feature engineering and normalization with aggregated features.

    Key design principles:
    - Aggregates correlated hydrophobicity scales (13 -> 2)
    - Aggregates secondary structure propensities (9 -> 3)
    - Converts absolute coordinates to relative (centroid-based)
    - Adds spherical coordinate representation
    - Uses LabelEncoder for categorical (element, residue) -> embedding indices
    """

    # Categorical columns (excluded from numerical processing)
    CATEGORICAL_COLS = ['original_index', 'chain_id', 'residue_name', 'atom_type', 'element_symbol', 'Unnamed: 0']

    # Core features to keep (19 features)
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

    # Hydrophobicity columns to aggregate (highly correlated, 13 scales)
    HYDROPHOBICITY_COLS = [
        'expasy:hphob_eisenberg', 'expasy:hphob_sweet', 'expasy:hphob_woods',
        'expasy:hphob_doolittle', 'expasy:hphob_leo', 'expasy:hphob_fauchere',
        'expasy:hphob_guy', 'expasy:hphob_janin', 'expasy:hphob_roseman',
        'expasy:hphob_tanford', 'expasy:hphob_parker', 'expasy:hphob_chothia',
        'expasy:hphob_rose'
    ]

    # Secondary structure propensity columns to aggregate (3 each)
    HELIX_COLS = ['expasy:alpha_helixfasman', 'expasy:alpha_helixroux', 'expasy:alpha_helixlevitt']
    SHEET_COLS = ['expasy:beta_sheetfasman', 'expasy:beta_sheetroux', 'expasy:beta_sheetlevitt']
    TURN_COLS = ['expasy:beta_turnfasman', 'expasy:beta_turnroux', 'expasy:beta_turnlevitt']

    # Standard elements in proteins
    STANDARD_ELEMENTS = ['C', 'N', 'O', 'S', 'OTHER']

    # Standard amino acids (20 standard + OTHER)
    STANDARD_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                         'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                         'THR', 'TRP', 'TYR', 'VAL', 'OTHER']

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.element_encoder = LabelEncoder()
        self.residue_encoder = LabelEncoder()
        self.fitted = False

        self.feature_names: List[str] = []
        self.element_classes: List[str] = []
        self.residue_classes: List[str] = []

    def _engineer_node_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to node dataframe.

        Returns new dataframe with 31 engineered features:
        - 19 core features
        - 2 aggregated hydrophobicity (mean, std)
        - 3 aggregated secondary structure
        - 7 geometric features
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

    def fit_nodes(self, node_dfs: List[pd.DataFrame]) -> 'AggregatedFeatureNormalizer':
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

        # Fit element encoder using known categories
        self.element_encoder.fit(self.STANDARD_ELEMENTS)
        self.element_classes = list(self.element_encoder.classes_)

        # Fit residue encoder using known categories
        self.residue_encoder.fit(self.STANDARD_RESIDUES)
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
            raise RuntimeError("AggregatedFeatureNormalizer not fitted. Call fit_nodes() first.")

        # Engineer features
        engineered_df = self._engineer_node_features(df)

        # Normalize numerical features
        features = engineered_df.values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features_normalized = self.node_scaler.transform(features)

        # Encode elements (handle unknown elements)
        elements = df['element_symbol'].values
        element_indices = []
        other_idx = self.element_encoder.transform(['OTHER'])[0]
        for elem in elements:
            if elem in self.element_classes:
                element_indices.append(self.element_encoder.transform([elem])[0])
            else:
                element_indices.append(other_idx)
        element_indices = np.array(element_indices)

        # Encode residues (handle unknown residues)
        residues = df['residue_name'].values
        residue_indices = []
        other_res_idx = self.residue_encoder.transform(['OTHER'])[0]
        for res in residues:
            if res in self.residue_classes:
                residue_indices.append(self.residue_encoder.transform([res])[0])
            else:
                residue_indices.append(other_res_idx)
        residue_indices = np.array(residue_indices)

        return (
            torch.tensor(features_normalized, dtype=torch.float32),
            torch.tensor(element_indices, dtype=torch.long),
            torch.tensor(residue_indices, dtype=torch.long)
        )

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
                'element_encoder': self.element_encoder,
                'residue_encoder': self.residue_encoder,
                'feature_names': self.feature_names,
                'element_classes': self.element_classes,
                'residue_classes': self.residue_classes,
                'fitted': self.fitted
            }, f)

    def load(self, path: Path) -> 'AggregatedFeatureNormalizer':
        """Load normalizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.node_scaler = data['node_scaler']
            self.element_encoder = data['element_encoder']
            self.residue_encoder = data.get('residue_encoder', LabelEncoder())
            self.feature_names = data['feature_names']
            self.element_classes = data['element_classes']
            self.residue_classes = data.get('residue_classes', self.STANDARD_RESIDUES)
            self.fitted = data['fitted']
        return self


def get_aggregated_feature_dimensions(element_embed_dim: int = 8,
                                       residue_embed_dim: int = 11) -> Dict[str, int]:
    """
    Get the dimensions of each feature group for aggregated transforms.

    Args:
        element_embed_dim: Dimension of element embedding
        residue_embed_dim: Dimension of residue embedding

    Returns:
        Dictionary with feature group names and their dimensions
    """
    # Core features: 19
    # Hydrophobicity aggregates: 2 (mean, std)
    # Secondary structure: 3 (helix, sheet, turn)
    # Geometric: 7 (dist_from_centroid, rel_x/y/z, spherical_r/theta/phi)
    numerical_count = 19 + 2 + 3 + 7  # = 31

    total = numerical_count + element_embed_dim + residue_embed_dim

    return {
        'numerical': numerical_count,
        'element_embed': element_embed_dim,
        'residue_embed': residue_embed_dim,
        'num_elements': len(AggregatedFeatureNormalizer.STANDARD_ELEMENTS),
        'num_residues': len(AggregatedFeatureNormalizer.STANDARD_RESIDUES),
        'total': total
    }


if __name__ == '__main__':
    # Print feature dimensions for verification
    print("=" * 60)
    print("Aggregated Feature Dimensions")
    print("=" * 60)

    dims = get_aggregated_feature_dimensions()
    print(f"\nDefault configuration (8 element embed, 11 residue embed):")
    print(f"  Numerical: {dims['numerical']}")
    print(f"  Element embedding: {dims['element_embed']} (vocab size: {dims['num_elements']})")
    print(f"  Residue embedding: {dims['residue_embed']} (vocab size: {dims['num_residues']})")
    print(f"  TOTAL: {dims['total']}")  # Should be 50
