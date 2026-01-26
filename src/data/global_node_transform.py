"""
Global Node Transform for Protein Graphs

Adds a virtual global node to each protein graph that:
1. Is connected to all atoms bidirectionally
2. Contains aggregated global features of the protein
3. Helps the network learn global context and normalization

This addresses the issue where local predictions lack global context,
causing outliers (e.g., predicting exposure=1.5 when target=0).
"""

import torch
import numpy as np
from torch_geometric.data import Data


class AddGlobalNode:
    """
    Transform that adds a global virtual node to protein graphs.

    The global node:
    - Contains aggregated statistics (mean, std, min, max) of key features
    - Is connected to all atoms bidirectionally
    - Does NOT have a prediction target (mask applied during training)

    This gives the network global context about:
    - Overall protein size/composition
    - Distribution of exposure values (during training)
    - Global chemical properties

    Args:
        feature_aggregation_mode: Which features to aggregate
            - 'statistical': mean/std/min/max of numerical features
            - 'full': include categorical distributions
        include_target_stats: Whether to include target statistics in global node
            (only for training, helps network learn scale/distribution)
    """

    def __init__(self, aggregation: str = 'mean'):
        """
        Args:
            aggregation: Type of aggregation ('mean', 'max', 'sum')
                - 'mean': Average pooling (default, most balanced)
                - 'max': Max pooling (emphasizes extremes)
                - 'sum': Sum pooling (sensitive to protein size)
        """
        self.aggregation = aggregation

    def __call__(self, data: Data) -> Data:
        """
        Add global node to the protein graph.

        Args:
            data: PyG Data object with x, edge_index, edge_attr, y

        Returns:
            Modified Data object with:
            - x: (N+1, F) features (N atoms + 1 global node)
            - edge_index: Original edges + global connections
            - edge_attr: Original edge features + global edge features
            - y: Original targets + None/mask for global node
            - global_node_mask: Boolean tensor indicating global node
        """
        num_atoms = data.num_nodes
        num_features = data.x.size(1)

        # 1. Compute global node features
        global_features = self._compute_global_features(data)

        # Ensure global features have same dimension as atom features
        if global_features.size(0) < num_features:
            # Pad with zeros if needed
            padding = torch.zeros(num_features - global_features.size(0))
            global_features = torch.cat([global_features, padding])
        elif global_features.size(0) > num_features:
            # Truncate if too long (shouldn't happen)
            global_features = global_features[:num_features]

        # Reshape to (1, F)
        global_features = global_features.unsqueeze(0)

        # 2. Add global node to node features
        # Shape: (N, F) + (1, F) = (N+1, F)
        data.x = torch.cat([data.x, global_features], dim=0)

        # 3. Create edges between global node and all atoms
        # Global node index = N (last node)
        global_idx = num_atoms

        # Edges: global -> atoms AND atoms -> global
        global_to_atoms = torch.stack([
            torch.full((num_atoms,), global_idx, dtype=torch.long),  # src: all global
            torch.arange(num_atoms, dtype=torch.long)                 # dst: all atoms
        ], dim=0)

        atoms_to_global = torch.stack([
            torch.arange(num_atoms, dtype=torch.long),                # src: all atoms
            torch.full((num_atoms,), global_idx, dtype=torch.long)   # dst: all global
        ], dim=0)

        # Concatenate with original edges
        data.edge_index = torch.cat([
            data.edge_index,
            global_to_atoms,
            atoms_to_global
        ], dim=1)

        # 4. Create edge features for global connections
        # Use a special "global connection" indicator
        num_edge_features = data.edge_attr.size(1) if data.edge_attr is not None else 12

        # Global edge features: zeros except for a special flag
        global_edge_features = torch.zeros(num_atoms * 2, num_edge_features)
        # Set last feature as "global connection" indicator
        global_edge_features[:, -1] = 1.0  # Mark as global edge

        if data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, global_edge_features], dim=0)
        else:
            data.edge_attr = global_edge_features

        # 5. Add mask for global node target
        # Target for global node = -1 (will be masked during training)
        global_target = torch.tensor([-1.0])
        data.y = torch.cat([data.y, global_target])

        # 6. Add global node mask (useful for filtering during prediction)
        global_node_mask = torch.zeros(num_atoms + 1, dtype=torch.bool)
        global_node_mask[-1] = True
        data.global_node_mask = global_node_mask

        # Update num_nodes
        data.num_nodes = num_atoms + 1

        return data

    def _compute_global_features(self, data: Data) -> torch.Tensor:
        """
        Compute aggregated global features from all atoms.

        Strategy: Feature-wise aggregation to maintain same structure as atom features.

        For each feature i in [0, F):
            global_node[i] = aggregation(atoms[:, i])

        This ensures global node has SAME feature format as atoms (92 features).

        Interpretation by feature type:
        - Numerical features (0-23):
            * mean: average biochemical property
            * max: extreme/dominant property
            * sum: total amount (scales with protein size)

        - Categorical one-hot (24-80):
            * mean: distribution/prevalence (e.g., 0.3 = 30% are CA atoms)
            * max: presence indicator (1.0 if type exists, 0.0 otherwise)
            * sum: count of each type

        - Geometric features (81-87):
            * mean: average 3D structure characteristics
            * max: maximum distances/positions
            * sum: total (less meaningful for distances)

        - Backbone angles (88-91):
            * mean: average torsion angles (valid for sin/cos)
            * max: extreme angles
            * sum: not physically meaningful

        Returns:
            Tensor of shape (F,) with same dimension as atom features
        """
        x = data.x  # (N, F)

        if self.aggregation == 'mean':
            global_features = x.mean(dim=0)  # (F,)
        elif self.aggregation == 'max':
            global_features = x.max(dim=0)[0]  # (F,)
        elif self.aggregation == 'sum':
            global_features = x.sum(dim=0)  # (F,)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return global_features

    def __repr__(self):
        return f'{self.__class__.__name__}(aggregation={self.aggregation})'


class AddGlobalNodeMax:
    """
    Global node variant using max pooling aggregation.

    Max pooling emphasizes extreme values, which might be useful for
    capturing protein-level features like maximum exposure or burial.

    Uses: global_features[i] = max(atoms[:, i]) for all features.
    """

    def __call__(self, data: Data) -> Data:
        """Add global node with max-pooled features."""
        num_atoms = data.num_nodes

        # Max pooling: global_features[i] = max(atoms[:, i])
        global_features = data.x.max(dim=0)[0]  # (F,)
        global_features = global_features.unsqueeze(0)  # (1, F)

        # Add global node to features
        data.x = torch.cat([data.x, global_features], dim=0)

        # Create global edges (bidirectional)
        global_idx = num_atoms

        global_to_atoms = torch.stack([
            torch.full((num_atoms,), global_idx, dtype=torch.long),
            torch.arange(num_atoms, dtype=torch.long)
        ], dim=0)

        atoms_to_global = torch.stack([
            torch.arange(num_atoms, dtype=torch.long),
            torch.full((num_atoms,), global_idx, dtype=torch.long)
        ], dim=0)

        data.edge_index = torch.cat([
            data.edge_index,
            global_to_atoms,
            atoms_to_global
        ], dim=1)

        # Global edge features
        num_edge_features = data.edge_attr.size(1) if data.edge_attr is not None else 12
        global_edge_features = torch.zeros(num_atoms * 2, num_edge_features)
        global_edge_features[:, -1] = 1.0  # Mark as global edge

        if data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, global_edge_features], dim=0)

        # Global node target (masked)
        data.y = torch.cat([data.y, torch.tensor([-1.0])])

        # Global node mask
        global_node_mask = torch.zeros(num_atoms + 1, dtype=torch.bool)
        global_node_mask[-1] = True
        data.global_node_mask = global_node_mask

        data.num_nodes = num_atoms + 1

        return data


if __name__ == '__main__':
    # Test the transform
    print("Testing Global Node Transform...")
    print("=" * 60)

    # Create dummy data
    num_atoms = 100
    num_features = 93

    x = torch.randn(num_atoms, num_features)
    edge_index = torch.randint(0, num_atoms, (2, 500))
    edge_attr = torch.randn(500, 12)
    y = torch.randn(num_atoms)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    print(f"Original data:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge features: {data.edge_attr.shape}")
    print(f"  Targets: {data.y.shape}")

    # Apply transform (test mean aggregation)
    transform = AddGlobalNode(aggregation='mean')
    data_transformed = transform(data)

    print(f"\nAfter adding global node:")
    print(f"  Nodes: {data_transformed.num_nodes} (+1)")
    print(f"  Edges: {data_transformed.edge_index.size(1)} (+{num_atoms * 2})")
    print(f"  Node features: {data_transformed.x.shape}")
    print(f"  Edge features: {data_transformed.edge_attr.shape}")
    print(f"  Targets: {data_transformed.y.shape}")
    print(f"  Global node mask: {data_transformed.global_node_mask.sum().item()} node(s)")

    # Verify global node connections
    global_idx = num_atoms
    global_edges = (data_transformed.edge_index[0] == global_idx) | (data_transformed.edge_index[1] == global_idx)
    print(f"\nGlobal node (idx={global_idx}) connections:")
    print(f"  Total edges involving global node: {global_edges.sum().item()}")
    print(f"  Expected: {num_atoms * 2} (bidirectional to all atoms)")

    # Check global edge features
    global_edge_mask = data_transformed.edge_attr[:, -1] == 1.0
    print(f"  Edges marked as global: {global_edge_mask.sum().item()}")

    # Verify global node features
    print(f"\nGlobal node features:")
    global_node_features = data_transformed.x[-1]  # Last node = global node
    print(f"  Feature vector shape: {global_node_features.shape}")
    print(f"  Same as atom features: {global_node_features.shape == data_transformed.x[0].shape}")
    print(f"  First 5 features: {global_node_features[:5]}")
    print(f"  Expected: mean of all atoms for each feature")

    # Compare with manual mean calculation
    manual_mean = x.mean(dim=0)
    matches = torch.allclose(global_node_features, manual_mean, rtol=1e-5)
    print(f"  Matches manual mean: {matches}")

    print("\n" + "=" * 60)
    print("Global node transform test completed!")
