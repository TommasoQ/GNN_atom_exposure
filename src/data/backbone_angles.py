"""
Backbone Dihedral Angle Calculation (φ/ψ)

Calculates phi and psi backbone torsion angles from atomic coordinates.
These angles determine secondary structure and are informative for exposure prediction.

φ (phi): dihedral angle C(i-1) - N(i) - Cα(i) - C(i)
ψ (psi): dihedral angle N(i) - Cα(i) - C(i) - N(i+1)

Output: sin/cos encoding to avoid discontinuity at ±180°
  - sin(φ), cos(φ), sin(ψ), cos(ψ) → 4 features per atom
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral angle between 4 points.

    The dihedral angle is the angle between the planes defined by
    (p1, p2, p3) and (p2, p3, p4).

    Args:
        p1, p2, p3, p4: 3D coordinates as numpy arrays

    Returns:
        Dihedral angle in radians [-π, π]
    """
    # Vectors along the bonds
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        # Degenerate case (collinear atoms)
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Calculate angle using atan2 for correct quadrant
    b2_normalized = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_normalized)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)


def extract_backbone_angles(nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Extract phi/psi backbone angles for each atom in the protein.

    Each atom inherits the phi/psi angles of its residue.
    Terminal residues and residues with missing backbone atoms get (0, 0).

    Encoding strategy:
    - Valid angles: (sin(angle), cos(angle)) → norm = 1
    - Invalid angles: (0, 0) → norm = 0

    This allows the model to learn that norm=1 indicates valid angle data,
    while norm=0 indicates missing/invalid angles (terminal residues).

    Args:
        nodes_df: DataFrame with columns:
            - chain_id: chain identifier
            - residue_number: residue sequence number
            - atom_type: atom name (N, CA, C, etc.)
            - x_coord, y_coord, z_coord: atomic coordinates

    Returns:
        np.ndarray of shape (n_atoms, 4): [sin_phi, cos_phi, sin_psi, cos_psi]
        - Valid angles: sin²+cos² = 1
        - Invalid angles: (0, 0) with norm = 0
    """
    n_atoms = len(nodes_df)
    # Initialize with zeros - (0,0) indicates invalid/missing angle
    angles = np.zeros((n_atoms, 4), dtype=np.float32)

    # Extract coordinates
    coords = nodes_df[['x_coord', 'y_coord', 'z_coord']].values

    # Build residue information
    # Group atoms by (chain_id, residue_number)
    residue_groups = nodes_df.groupby(['chain_id', 'residue_number'])

    # For each chain, process residues in order
    for chain_id in nodes_df['chain_id'].unique():
        chain_mask = nodes_df['chain_id'] == chain_id
        chain_df = nodes_df[chain_mask]

        # Get sorted unique residue numbers for this chain
        residue_numbers = sorted(chain_df['residue_number'].unique())

        # Build backbone atom lookup: {(chain, resnum): {'N': idx, 'CA': idx, 'C': idx}}
        backbone_lookup = {}
        for resnum in residue_numbers:
            res_mask = (nodes_df['chain_id'] == chain_id) & (nodes_df['residue_number'] == resnum)
            res_df = nodes_df[res_mask]

            backbone_atoms = {}
            for atom_type in ['N', 'CA', 'C']:
                atom_mask = res_df['atom_type'] == atom_type
                if atom_mask.any():
                    # Get the index in the original dataframe
                    idx = res_df[atom_mask].index[0]
                    # Convert to positional index
                    pos_idx = nodes_df.index.get_loc(idx)
                    backbone_atoms[atom_type] = pos_idx

            if len(backbone_atoms) == 3:  # All backbone atoms present
                backbone_lookup[(chain_id, resnum)] = backbone_atoms

        # Calculate phi/psi for each residue
        # Use None to indicate invalid/missing angles
        residue_angles = {}  # {(chain, resnum): (phi, psi)} where None = invalid

        for i, resnum in enumerate(residue_numbers):
            key = (chain_id, resnum)

            if key not in backbone_lookup:
                residue_angles[key] = (None, None)
                continue

            phi = None  # None = couldn't compute (N-terminal or missing atoms)
            psi = None  # None = couldn't compute (C-terminal or missing atoms)

            # PHI: C(i-1) - N(i) - CA(i) - C(i)
            if i > 0:
                prev_resnum = residue_numbers[i - 1]
                prev_key = (chain_id, prev_resnum)

                if prev_key in backbone_lookup and key in backbone_lookup:
                    try:
                        c_prev = coords[backbone_lookup[prev_key]['C']]
                        n_curr = coords[backbone_lookup[key]['N']]
                        ca_curr = coords[backbone_lookup[key]['CA']]
                        c_curr = coords[backbone_lookup[key]['C']]

                        phi = calculate_dihedral(c_prev, n_curr, ca_curr, c_curr)
                    except (KeyError, IndexError):
                        phi = None

            # PSI: N(i) - CA(i) - C(i) - N(i+1)
            if i < len(residue_numbers) - 1:
                next_resnum = residue_numbers[i + 1]
                next_key = (chain_id, next_resnum)

                if key in backbone_lookup and next_key in backbone_lookup:
                    try:
                        n_curr = coords[backbone_lookup[key]['N']]
                        ca_curr = coords[backbone_lookup[key]['CA']]
                        c_curr = coords[backbone_lookup[key]['C']]
                        n_next = coords[backbone_lookup[next_key]['N']]

                        psi = calculate_dihedral(n_curr, ca_curr, c_curr, n_next)
                    except (KeyError, IndexError):
                        psi = None

            residue_angles[key] = (phi, psi)

        # Assign angles to all atoms in each residue
        for resnum in residue_numbers:
            key = (chain_id, resnum)
            phi, psi = residue_angles.get(key, (None, None))

            # Find all atoms in this residue
            res_mask = (nodes_df['chain_id'] == chain_id) & (nodes_df['residue_number'] == resnum)
            res_indices = np.where(res_mask)[0]

            # Assign sin/cos encoded angles
            # Valid angles: (sin, cos) with norm = 1
            # Invalid angles: (0, 0) with norm = 0
            if phi is not None:
                angles[res_indices, 0] = np.sin(phi)
                angles[res_indices, 1] = np.cos(phi)
            # else: stays (0, 0) from initialization

            if psi is not None:
                angles[res_indices, 2] = np.sin(psi)
                angles[res_indices, 3] = np.cos(psi)
            # else: stays (0, 0) from initialization

    return angles


def get_backbone_feature_names() -> list:
    """Return feature names for backbone angles."""
    return ['backbone_sin_phi', 'backbone_cos_phi', 'backbone_sin_psi', 'backbone_cos_psi']


if __name__ == '__main__':
    # Test on a sample protein
    import os

    # Load sample protein
    sample_dir = '../../dataset/sadic_data/142l'
    nodes_path = os.path.join(sample_dir, '142l__graphein__ATOM_nodes.csv')

    if os.path.exists(nodes_path):
        nodes_df = pd.read_csv(nodes_path, index_col=0)
        print(f"Loaded protein with {len(nodes_df)} atoms")
        print(f"Columns: {list(nodes_df.columns)}")
        print(f"Chains: {nodes_df['chain_id'].unique()}")
        print(f"Residues: {nodes_df['residue_number'].nunique()}")

        # Extract backbone angles
        angles = extract_backbone_angles(nodes_df)
        print(f"\nBackbone angles shape: {angles.shape}")
        print(f"Feature names: {get_backbone_feature_names()}")

        # Statistics
        print(f"\nAngle statistics:")
        for i, name in enumerate(get_backbone_feature_names()):
            print(f"  {name}: mean={angles[:, i].mean():.4f}, std={angles[:, i].std():.4f}")

        # Check for non-zero angles (should be most residues except termini)
        non_zero_phi = np.sum(np.abs(angles[:, 0]) > 0.01)
        non_zero_psi = np.sum(np.abs(angles[:, 2]) > 0.01)
        print(f"\nAtoms with non-zero phi: {non_zero_phi}/{len(nodes_df)}")
        print(f"Atoms with non-zero psi: {non_zero_psi}/{len(nodes_df)}")

        # Sample output
        print(f"\nSample angles (first 10 atoms):")
        for i in range(min(10, len(nodes_df))):
            row = nodes_df.iloc[i]
            print(f"  {row['residue_name']}{row['residue_number']}:{row['atom_type']} -> "
                  f"sin_phi={angles[i, 0]:.3f}, cos_phi={angles[i, 1]:.3f}, "
                  f"sin_psi={angles[i, 2]:.3f}, cos_psi={angles[i, 3]:.3f}")
    else:
        print(f"Sample file not found: {nodes_path}")
