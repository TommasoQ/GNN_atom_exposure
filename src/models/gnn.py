"""
Graph Neural Network Model for Atom Exposure Prediction

This module contains the MinimalGCN model - a simple but effective GCN
that achieves R² ~0.87-0.89 with only 5 geometric features and ~16K parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm


class MinimalGCN(nn.Module):
    """
    Minimal GCN model - no edge features, reduced architecture.

    Designed for use with minimal geometric features (5 features):
    - geom_contact_count_10A (dominant feature)
    - geom_dist_to_center
    - geom_radial_position
    - geom_3rd_nearest_dist
    - geom_std_dist

    Optionally supports dynamic global pooling for protein-level context.

    Args:
        in_channels (int): Number of input features (5 for minimal)
        hidden_channels (int): Number of hidden units (default 64)
        num_layers (int): Number of GCN layers (default 2)
        dropout (float): Dropout rate (default 0.2)
        use_global_pool (bool): Enable dynamic global pooling (default False)
        global_pool_type (str): Type of pooling - 'mean' or 'max' (default 'mean')
        feature_noise (float): Gaussian noise σ to add to features during training (default 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_global_pool: bool = False,
        global_pool_type: str = 'mean',
        feature_noise: float = 0.0
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_global_pool = use_global_pool
        self.global_pool_type = global_pool_type
        self.feature_noise = feature_noise

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GCN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])

        # Global pooling gates (if enabled)
        # Each gate learns how much protein-level context to inject per layer
        if use_global_pool:
            self.global_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Sigmoid()
                ) for _ in range(num_layers)
            ])
            # LayerNorm after global pooling injection for stability
            self.global_layer_norms = nn.ModuleList([
                LayerNorm(hidden_channels) for _ in range(num_layers)
            ])

        # Output layers
        self.out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        """
        Forward pass - ignores edge_attr since edge features have 0 importance.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Ignored (edge features have 0 importance)
            batch: Batch vector [num_nodes]
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Predicted atom exposure values [num_nodes]
        """
        # Create batch vector if not provided (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Add Gaussian noise during training for regularization
        if self.training and self.feature_noise > 0:
            noise = torch.randn_like(x) * self.feature_noise
            x = x + noise

        # Input projection
        x = F.relu(self.input_proj(x))

        # GCN layers with residual connections and optional global pooling
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_in = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_in

            # Dynamic global pooling: inject protein-level context
            if self.use_global_pool:
                # 1. Aggregate all nodes to protein-level representation
                if self.global_pool_type == 'mean':
                    global_repr = global_mean_pool(x, batch)  # (num_graphs, H)
                else:
                    global_repr = global_max_pool(x, batch)   # (num_graphs, H)

                # 2. Broadcast back to node level
                global_expanded = global_repr[batch]  # (num_nodes, H)

                # 3. Gated combination: learn how much global context to use
                gate_input = torch.cat([x, global_expanded], dim=-1)
                gate = self.global_gates[i](gate_input)  # (num_nodes, H), values in [0,1]

                # 4. Add gated global context
                x = x + gate * global_expanded

                # 5. Stabilize with LayerNorm
                x = self.global_layer_norms[i](x)

        # Output
        return self.out(x).squeeze(-1)


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create a model from configuration.

    Args:
        config (dict): Model configuration with keys:
            - in_channels: Number of input features
            - hidden_channels: Hidden layer size (default 64)
            - num_layers: Number of GCN layers (default 2)
            - dropout: Dropout rate (default 0.2)
            - use_global_pool: Enable global pooling (default False)
            - global_pool_type: 'mean' or 'max' (default 'mean')
            - feature_noise: Training noise σ (default 0.0)

    Returns:
        nn.Module: Initialized MinimalGCN model
    """
    return MinimalGCN(
        in_channels=config['in_channels'],
        hidden_channels=config.get('hidden_channels', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        use_global_pool=config.get('use_global_pool', False),
        global_pool_type=config.get('global_pool_type', 'mean'),
        feature_noise=config.get('feature_noise', 0.0)
    )
