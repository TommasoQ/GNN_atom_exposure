"""
Feature Engineering Module
Handles feature selection, categorical encoding, geometric features, and normalization
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle


# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Based on Phase 1 correlation analysis + Phase 3 feature reduction (14 features removed)
SELECTED_NUMERICAL_FEATURES = [
    # Core features (high correlation)
    'b_factor',                      # 0.63 correlation - strongest predictor!
    'hbond_donors',
    'hbond_acceptors',

    # Meiler descriptors (keep most relevant, dropped dim_6)
    'meiler:dim_1',
    'meiler:dim_4',                  # -0.41 correlation
    'meiler:dim_5',
    'meiler:dim_7',
    # Removed: meiler:dim_6 (r=0.83 with coilroux, weak target correlation)

    # Hydrophobicity scales (kept 3 most diverse, dropped janin/chothia/woods)
    'expasy:hphob_eisenberg',        # -0.45 (most established scale)
    'expasy:hphob_rose',             # -0.42 (best target correlation)
    'expasy:hphob_guy',              # +0.47 (different pattern, complementary)
    # Removed: hphob_janin (r=0.93 with eisenberg)
    # Removed: hphob_chothia (r=0.90 with eisenberg)
    # Removed: hphob_woods (r=-0.93 with transmembranetendency)

    # Structural propensities (directly relevant)
    'expasy:buriedresidues',         # Directly relevant!
    'expasy:accessibleresidues',     # Directly relevant!
    'expasy:averageburied',          # Directly relevant!
    'expasy:averageflexibility',
    'expasy:transmembranetendency',  # -0.41
    'expasy:totalbeta_strand',       # General beta measure
    'expasy:parallelbeta_strand',    # Complementary to total
    # Removed: antiparallelbeta_strand (r=0.95 with totalbeta_strand)

    # Polarity and molecular properties
    'expasy:polarityzimmerman',      # +0.35
    'expasy:polaritygrantham',       # +0.39
    'expasy:bulkiness',
    'expasy:ratioside',
    # Removed: isoelectric_points (r=0.91 with meiler:dim_5)
    # Removed: molecularweight (r=-0.012 target, redundant with residue type)
    # Removed: refractivity (r=0.92 with molecularweight)

    # Secondary structure propensities (kept turn features, dropped redundant sheet)
    'expasy:beta_turnfasman',
    'expasy:beta_turnroux',
    'expasy:coilroux',
    # Removed: beta_sheetfasman (r=0.95 with totalbeta_strand)
    # Removed: beta_sheetroux (r=0.97 with totalbeta_strand)
]

# Total: 24 numerical features (reduced from 34)


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================

# Standard atom types in proteins (most common)
STANDARD_ATOM_TYPES = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CD1', 'CD2',
                       'CG1', 'CG2', 'CE', 'CZ', 'OD1', 'OD2', 'OE1', 'OE2',
                       'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ', 'OG', 'OG1',
                       'SD', 'SG', 'CE1', 'CE2', 'CE3', 'OTHER']

# Standard elements in proteins (removed P - virtually no phosphorus in dataset)
STANDARD_ELEMENTS = ['C', 'N', 'O', 'S', 'OTHER']

# Standard amino acids (20 standard + OTHER catchall - OTHER has no data but needed for encoding)
STANDARD_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                     'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                     'THR', 'TRP', 'TYR', 'VAL', 'OTHER']


def one_hot_encode(values: pd.Series, categories: List[str],
                   unknown_category: str = 'OTHER') -> np.ndarray:
    """
    One-hot encode a categorical column.

    Args:
        values: Series of categorical values
        categories: List of valid categories (includes unknown_category)
        unknown_category: Category to use for unknown values

    Returns:
        One-hot encoded array of shape (n_samples, n_categories)
    """
    n_samples = len(values)
    n_categories = len(categories)
    encoded = np.zeros((n_samples, n_categories), dtype=np.float32)

    # Create category to index mapping
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    unknown_idx = cat_to_idx[unknown_category]

    # Encode each value
    for i, val in enumerate(values):
        idx = cat_to_idx.get(val, unknown_idx)
        encoded[i, idx] = 1.0

    return encoded


def encode_atom_types(atom_types: pd.Series) -> np.ndarray:
    """One-hot encode atom types."""
    return one_hot_encode(atom_types, STANDARD_ATOM_TYPES, 'OTHER')


def encode_elements(elements: pd.Series) -> np.ndarray:
    """One-hot encode element symbols."""
    return one_hot_encode(elements, STANDARD_ELEMENTS, 'OTHER')


def encode_residues(residues: pd.Series) -> np.ndarray:
    """One-hot encode residue names."""
    return one_hot_encode(residues, STANDARD_RESIDUES, 'OTHER')


def encode_categorical_features(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Encode all categorical features from a nodes DataFrame.

    Args:
        nodes_df: DataFrame with columns 'atom_type', 'element_symbol', 'residue_name'

    Returns:
        Array of shape (n_atoms, n_categorical_features)
    """
    atom_encoded = encode_atom_types(nodes_df['atom_type'])
    element_encoded = encode_elements(nodes_df['element_symbol'])
    residue_encoded = encode_residues(nodes_df['residue_name'])

    # Concatenate all categorical features
    categorical_features = np.concatenate([
        atom_encoded,
        element_encoded,
        residue_encoded
    ], axis=1)

    return categorical_features


# ============================================================================
# GEOMETRIC FEATURES
# ============================================================================

def compute_geometric_features(coords: np.ndarray,
                                edge_index: np.ndarray,
                                edge_distances: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute rotation/translation invariant geometric features.

    Args:
        coords: Array of shape (n_atoms, 3) with xyz coordinates
        edge_index: Array of shape (2, n_edges) with edge connectivity
        edge_distances: Optional array of shape (n_edges,) with precomputed distances

    Returns:
        Array of shape (n_atoms, n_geometric_features)
    """
    n_atoms = coords.shape[0]

    # Initialize features array (7 features total - removed duplicate geom_nearest_dist)
    geom_features = np.zeros((n_atoms, 7), dtype=np.float32)

    # Compute protein center of mass
    center_of_mass = coords.mean(axis=0)

    # For each atom, compute geometric features
    for i in range(n_atoms):
        # Find neighbors
        neighbors_mask = (edge_index[0] == i)
        neighbor_indices = edge_index[1][neighbors_mask]

        if len(neighbor_indices) == 0:
            continue

        # Get neighbor coordinates
        neighbor_coords = coords[neighbor_indices]

        # Compute distances to neighbors
        if edge_distances is not None:
            distances = edge_distances[neighbors_mask]
        else:
            distances = np.linalg.norm(neighbor_coords - coords[i], axis=1)

        # Feature 1: Mean distance to neighbors
        geom_features[i, 0] = distances.mean()

        # Feature 2: Min distance to neighbor
        geom_features[i, 1] = distances.min()

        # Feature 3: Max distance to neighbor
        geom_features[i, 2] = distances.max()

        # Feature 4: Std of distances
        geom_features[i, 3] = distances.std() if len(distances) > 1 else 0.0

        # Feature 5: Distance to 3rd nearest neighbor (if exists)
        # NOTE: Removed geom_nearest_dist (Feature 5) - perfect duplicate of geom_min_dist (r=1.0)
        sorted_dist = np.sort(distances)
        geom_features[i, 4] = sorted_dist[min(2, len(sorted_dist)-1)]

        # Feature 6: Distance to protein center of mass
        geom_features[i, 5] = np.linalg.norm(coords[i] - center_of_mass)

        # Feature 7: Normalized radial position (0 = center, 1 = surface)
        max_distance_from_center = np.linalg.norm(coords - center_of_mass, axis=1).max()
        geom_features[i, 6] = geom_features[i, 5] / (max_distance_from_center + 1e-8)

    return geom_features


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

class FeatureNormalizer:
    """
    Normalizes numerical features to zero mean and unit variance.
    Only normalizes numerical features, not categorical or geometric.
    """

    def __init__(self):
        self.numerical_mean = None
        self.numerical_std = None
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """
        Compute mean and std from training data.

        Args:
            features: Array of shape (n_samples, n_features)
        """
        self.numerical_mean = features.mean(axis=0)
        self.numerical_std = features.std(axis=0)

        # Avoid division by zero
        self.numerical_std[self.numerical_std < 1e-8] = 1.0

        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization: (x - mean) / std

        Args:
            features: Array of shape (n_samples, n_features)

        Returns:
            Normalized features
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        return (features - self.numerical_mean) / self.numerical_std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def save(self, path: str):
        """Save normalization statistics to file."""
        stats = {
            'mean': self.numerical_mean,
            'std': self.numerical_std,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(stats, f)

    def load(self, path: str):
        """Load normalization statistics from file."""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        self.numerical_mean = stats['mean']
        self.numerical_std = stats['std']
        self.is_fitted = stats['is_fitted']


# ============================================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_all_features(nodes_df: pd.DataFrame,
                         edge_index: np.ndarray,
                         edge_distances: Optional[np.ndarray] = None,
                         normalizer: Optional[FeatureNormalizer] = None,
                         normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and combine all features: numerical, categorical, and geometric.

    Args:
        nodes_df: DataFrame with node features
        edge_index: Edge connectivity
        edge_distances: Optional precomputed edge distances
        normalizer: Optional fitted normalizer for numerical features
        normalize: Whether to normalize numerical features

    Returns:
        Tuple of (features_array, feature_names)
    """
    # 1. Extract selected numerical features
    numerical_features = nodes_df[SELECTED_NUMERICAL_FEATURES].values.astype(np.float32)

    # Normalize if requested
    if normalize:
        if normalizer is None:
            # Create and fit new normalizer
            normalizer = FeatureNormalizer()
            numerical_features = normalizer.fit_transform(numerical_features)
        else:
            # Use provided normalizer
            numerical_features = normalizer.transform(numerical_features)

    # 2. Extract categorical features
    categorical_features = encode_categorical_features(nodes_df)

    # 3. Extract geometric features
    coords = nodes_df[['x_coord', 'y_coord', 'z_coord']].values
    geometric_features = compute_geometric_features(coords, edge_index, edge_distances)

    # 4. Concatenate all features
    all_features = np.concatenate([
        numerical_features,
        categorical_features,
        geometric_features
    ], axis=1)

    # 5. Generate feature names
    numerical_names = SELECTED_NUMERICAL_FEATURES
    categorical_names = (
        [f'atom_{cat}' for cat in STANDARD_ATOM_TYPES] +
        [f'element_{cat}' for cat in STANDARD_ELEMENTS] +
        [f'residue_{cat}' for cat in STANDARD_RESIDUES]
    )
    geometric_names = [
        'geom_mean_dist', 'geom_min_dist', 'geom_max_dist', 'geom_std_dist',
        'geom_3rd_nearest_dist',  # Removed geom_nearest_dist (duplicate of geom_min_dist)
        'geom_dist_to_center', 'geom_radial_position'
    ]
    feature_names = numerical_names + categorical_names + geometric_names

    return all_features, feature_names


def get_feature_dimensions() -> Dict[str, int]:
    """
    Get the dimensions of each feature group.

    Returns:
        Dictionary with feature group names and their dimensions
    """
    return {
        'numerical': len(SELECTED_NUMERICAL_FEATURES),
        'atom_types': len(STANDARD_ATOM_TYPES),
        'elements': len(STANDARD_ELEMENTS),
        'residues': len(STANDARD_RESIDUES),
        'geometric': 7,  # Reduced from 8 (removed geom_nearest_dist duplicate)
        'total': (len(SELECTED_NUMERICAL_FEATURES) +  # 24
                 len(STANDARD_ATOM_TYPES) +           # 31
                 len(STANDARD_ELEMENTS) +             # 5 (includes OTHER)
                 len(STANDARD_RESIDUES) +             # 21 (includes OTHER)
                 7)                                   # = 88 total
    }


if __name__ == '__main__':
    # Print feature dimensions for verification
    dims = get_feature_dimensions()
    print("Feature Dimensions:")
    print(f"  Numerical: {dims['numerical']}")
    print(f"  Atom types: {dims['atom_types']}")
    print(f"  Elements: {dims['elements']}")
    print(f"  Residues: {dims['residues']}")
    print(f"  Geometric: {dims['geometric']}")
    print(f"  TOTAL: {dims['total']}")
