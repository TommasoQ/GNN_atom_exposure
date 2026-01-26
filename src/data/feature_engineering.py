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

# Import backbone angle calculation
try:
    from .backbone_angles import extract_backbone_angles, get_backbone_feature_names
except ImportError:
    from backbone_angles import extract_backbone_angles, get_backbone_feature_names


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


# Reduced numerical features (further redundancy removal based on correlation analysis)
# Dropped: transmembranetendency, meiler:dim_4, hphob_eisenberg, hphob_guy
# (all |r| > 0.90 correlated with hphob_rose or polaritygrantham)
REDUCED_NUMERICAL_FEATURES = [
    # Core features (high correlation)
    'b_factor',
    'hbond_donors',
    'hbond_acceptors',

    # Meiler descriptors (dropped dim_4 - r=0.94 with transmembranetendency)
    'meiler:dim_1',
    'meiler:dim_5',
    'meiler:dim_7',

    # Hydrophobicity - keep only hphob_rose (best target correlation in cluster)
    'expasy:hphob_rose',
    # Dropped: hphob_eisenberg (r=0.90 with transmembranetendency)
    # Dropped: hphob_guy (r=-0.93 with hphob_rose)

    # Structural propensities
    'expasy:buriedresidues',
    'expasy:accessibleresidues',
    'expasy:averageburied',
    'expasy:averageflexibility',
    'expasy:totalbeta_strand',
    'expasy:parallelbeta_strand',
    # Dropped: transmembranetendency (r=-0.95 with polaritygrantham)

    # Polarity - keep polaritygrantham (representative of polarity cluster)
    'expasy:polarityzimmerman',
    'expasy:polaritygrantham',
    'expasy:bulkiness',
    'expasy:ratioside',

    # Secondary structure propensities
    'expasy:beta_turnfasman',
    'expasy:beta_turnroux',
    'expasy:coilroux',
]
# Total: 20 numerical features (reduced from 24)


# Reduced geometric feature indices (keep: min_dist, std_dist, radial_position)
# Original indices: 0=mean_dist, 1=min_dist, 2=max_dist, 3=std_dist,
#                   4=3rd_nearest_dist, 5=dist_to_center, 6=radial_position
# Keep: min_dist (1), std_dist (3), radial_position (6)
# Dropped: mean_dist, max_dist, 3rd_nearest_dist (all |r| > 0.90 correlated)
#          dist_to_center (r=0.89 with radial_position)
REDUCED_GEOMETRIC_INDICES = [1, 3, 6]
REDUCED_GEOMETRIC_NAMES = ['geom_min_dist', 'geom_std_dist', 'geom_radial_position']


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


def encode_categorical_features(nodes_df: pd.DataFrame,
                                 include_atom_type: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Encode categorical features from a nodes DataFrame.

    Args:
        nodes_df: DataFrame with columns 'atom_type', 'element_symbol', 'residue_name'
        include_atom_type: Whether to include atom type one-hot encoding (31 features)

    Returns:
        Tuple of (encoded features array, feature names list)
    """
    features_list = []
    names_list = []

    if include_atom_type:
        atom_encoded = encode_atom_types(nodes_df['atom_type'])
        features_list.append(atom_encoded)
        names_list.extend([f'atom_{cat}' for cat in STANDARD_ATOM_TYPES])

    element_encoded = encode_elements(nodes_df['element_symbol'])
    features_list.append(element_encoded)
    names_list.extend([f'element_{cat}' for cat in STANDARD_ELEMENTS])

    residue_encoded = encode_residues(nodes_df['residue_name'])
    features_list.append(residue_encoded)
    names_list.extend([f'residue_{cat}' for cat in STANDARD_RESIDUES])

    categorical_features = np.concatenate(features_list, axis=1)

    return categorical_features, names_list


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
        Array of shape (n_atoms, 8) with geometric features
    """
    from scipy.spatial.distance import cdist

    n_atoms = coords.shape[0]

    # Initialize features array (8 features total)
    geom_features = np.zeros((n_atoms, 8), dtype=np.float32)

    # Compute protein center of mass
    center_of_mass = coords.mean(axis=0)

    # Compute pairwise distances for contact_count_10A (Feature 8)
    # Using cdist for efficiency - computes all pairwise distances at once
    pairwise_distances = cdist(coords, coords, metric='euclidean')

    # Feature 8: Contact count at 10Å (count of atoms within 10Å, excluding self)
    # CRITICAL FEATURE: Direct measure of burial/exposure
    # Subtract 1 to exclude self-distance (which is 0)
    contact_counts = (pairwise_distances < 10.0).sum(axis=1) - 1
    geom_features[:, 7] = contact_counts.astype(np.float32)

    # For each atom, compute remaining geometric features
    for i in range(n_atoms):
        # Find neighbors from graph structure
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
                         normalize: bool = True,
                         use_reduced_features: bool = False,
                         include_atom_type: bool = True,
                         include_geometric: bool = True,
                         include_backbone_angles: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and combine all features: numerical, categorical, geometric, and backbone angles.

    Args:
        nodes_df: DataFrame with node features
        edge_index: Edge connectivity
        edge_distances: Optional precomputed edge distances
        normalizer: Optional fitted normalizer for numerical features
        normalize: Whether to normalize numerical features
        use_reduced_features: Use reduced numerical/geometric features (removes redundancy)
        include_atom_type: Include atom type one-hot encoding (31 features)
        include_geometric: Include geometric features
        include_backbone_angles: Include backbone dihedral angles (phi/psi as sin/cos, 4 features)

    Returns:
        Tuple of (features_array, feature_names)
    """
    features_list = []
    feature_names = []

    # 1. Extract numerical features
    numerical_feature_list = REDUCED_NUMERICAL_FEATURES if use_reduced_features else SELECTED_NUMERICAL_FEATURES
    numerical_features = nodes_df[numerical_feature_list].values.astype(np.float32)

    # Normalize if requested
    if normalize:
        if normalizer is None:
            # Create and fit new normalizer
            normalizer = FeatureNormalizer()
            numerical_features = normalizer.fit_transform(numerical_features)
        else:
            # Use provided normalizer
            numerical_features = normalizer.transform(numerical_features)

    features_list.append(numerical_features)
    feature_names.extend(numerical_feature_list)

    # 2. Extract categorical features
    categorical_features, categorical_names = encode_categorical_features(
        nodes_df, include_atom_type=include_atom_type
    )
    features_list.append(categorical_features)
    feature_names.extend(categorical_names)

    # 3. Extract geometric features
    if include_geometric:
        coords = nodes_df[['x_coord', 'y_coord', 'z_coord']].values
        geometric_features = compute_geometric_features(coords, edge_index, edge_distances)

        if use_reduced_features:
            # Keep only non-redundant geometric features
            geometric_features = geometric_features[:, REDUCED_GEOMETRIC_INDICES]
            features_list.append(geometric_features)
            feature_names.extend(REDUCED_GEOMETRIC_NAMES)
        else:
            features_list.append(geometric_features)
            full_geometric_names = [
                'geom_mean_dist', 'geom_min_dist', 'geom_max_dist', 'geom_std_dist',
                'geom_3rd_nearest_dist', 'geom_dist_to_center', 'geom_radial_position',
                'geom_contact_count_10A'
            ]
            feature_names.extend(full_geometric_names)

    # 4. Extract backbone dihedral angles (phi/psi as sin/cos)
    if include_backbone_angles:
        backbone_features = extract_backbone_angles(nodes_df)
        features_list.append(backbone_features)
        feature_names.extend(get_backbone_feature_names())

    # 5. Concatenate all features
    all_features = np.concatenate(features_list, axis=1)

    return all_features, feature_names


def get_feature_dimensions(use_reduced_features: bool = False,
                           include_atom_type: bool = True,
                           include_geometric: bool = True,
                           include_backbone_angles: bool = False) -> Dict[str, int]:
    """
    Get the dimensions of each feature group based on configuration.

    Args:
        use_reduced_features: Use reduced numerical/geometric features
        include_atom_type: Include atom type one-hot encoding
        include_geometric: Include geometric features
        include_backbone_angles: Include backbone dihedral angles (phi/psi as sin/cos)

    Returns:
        Dictionary with feature group names and their dimensions
    """
    numerical_count = len(REDUCED_NUMERICAL_FEATURES) if use_reduced_features else len(SELECTED_NUMERICAL_FEATURES)
    atom_type_count = len(STANDARD_ATOM_TYPES) if include_atom_type else 0
    element_count = len(STANDARD_ELEMENTS)
    residue_count = len(STANDARD_RESIDUES)

    if include_geometric:
        geometric_count = len(REDUCED_GEOMETRIC_INDICES) if use_reduced_features else 8
    else:
        geometric_count = 0

    backbone_count = 4 if include_backbone_angles else 0  # sin_phi, cos_phi, sin_psi, cos_psi

    total = numerical_count + atom_type_count + element_count + residue_count + geometric_count + backbone_count

    return {
        'numerical': numerical_count,
        'atom_types': atom_type_count,
        'elements': element_count,
        'residues': residue_count,
        'geometric': geometric_count,
        'backbone_angles': backbone_count,
        'total': total
    }


if __name__ == '__main__':
    # Print feature dimensions for verification
    print("=" * 60)
    print("Feature Dimensions by Configuration")
    print("=" * 60)

    # Default (full features)
    dims = get_feature_dimensions()
    print(f"\nFull features (default):")
    print(f"  Numerical: {dims['numerical']}")
    print(f"  Atom types: {dims['atom_types']}")
    print(f"  Elements: {dims['elements']}")
    print(f"  Residues: {dims['residues']}")
    print(f"  Geometric: {dims['geometric']}")
    print(f"  TOTAL: {dims['total']}")

    # Reduced features with atom_type
    dims = get_feature_dimensions(use_reduced_features=True, include_atom_type=True)
    print(f"\nReduced features + atom_type:")
    print(f"  Numerical: {dims['numerical']}")
    print(f"  Atom types: {dims['atom_types']}")
    print(f"  Elements: {dims['elements']}")
    print(f"  Residues: {dims['residues']}")
    print(f"  Geometric: {dims['geometric']}")
    print(f"  TOTAL: {dims['total']}")

    # Reduced features without atom_type (matches colleague's 50)
    dims = get_feature_dimensions(use_reduced_features=True, include_atom_type=False)
    print(f"\nReduced features - atom_type (colleague's setup):")
    print(f"  Numerical: {dims['numerical']}")
    print(f"  Atom types: {dims['atom_types']}")
    print(f"  Elements: {dims['elements']}")
    print(f"  Residues: {dims['residues']}")
    print(f"  Geometric: {dims['geometric']}")
    print(f"  TOTAL: {dims['total']}")
