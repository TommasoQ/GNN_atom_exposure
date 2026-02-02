"""
Graph Neural Network Models for Atom Exposure Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GINEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm


class AtomExposureGNN(nn.Module):
    """
    Graph Neural Network for predicting atom exposure levels.

    Args:
        in_channels (int): Number of input node features (after embeddings if used)
        hidden_channels (int): Number of hidden units
        num_layers (int): Number of GNN layers
        dropout (float): Dropout rate
        conv_type (str): Type of graph convolution ('gcn', 'gat', 'gin', 'gine')
        edge_dim (int): Edge feature dimension (for gat, gine)
        use_embeddings (bool): Whether to use embedding layers for element/residue
        num_numerical (int): Number of numerical features (before embeddings)
        num_elements (int): Number of element types for embedding
        num_residues (int): Number of residue types for embedding
        element_embed_dim (int): Dimension of element embedding
        residue_embed_dim (int): Dimension of residue embedding
        use_global_pool (bool): Enable dynamic global pooling
        global_pool_type (str): Type of pooling ('mean', 'max', 'both')
        global_pool_layers (str): Where to inject ('every', 'middle', 'last')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = 'gcn',
        edge_dim: int = 1,
        use_embeddings: bool = False,
        num_numerical: int = 31,
        num_elements: int = 5,
        num_residues: int = 21,
        element_embed_dim: int = 8,
        residue_embed_dim: int = 11,
        use_global_pool: bool = False,
        global_pool_type: str = 'mean',
        global_pool_layers: str = 'every'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim

        # Global pooling configuration
        self.use_global_pool = use_global_pool
        self.global_pool_type = global_pool_type
        self.global_pool_layers = global_pool_layers

        # Embedding configuration
        self.use_embeddings = use_embeddings
        self.num_numerical = num_numerical
        self.element_embed_dim = element_embed_dim
        self.residue_embed_dim = residue_embed_dim

        # Create embedding layers if needed
        if use_embeddings:
            self.element_embedding = nn.Embedding(num_elements, element_embed_dim)
            self.residue_embedding = nn.Embedding(num_residues, residue_embed_dim)
            # Actual input to projection is numerical + embeddings
            actual_in_channels = num_numerical + element_embed_dim + residue_embed_dim
        else:
            self.element_embedding = None
            self.residue_embedding = None
            actual_in_channels = in_channels

        # Input projection
        self.input_proj = nn.Linear(actual_in_channels, hidden_channels)

        # Global pooling gates (if enabled)
        if use_global_pool:
            # Determine how many gates we need
            if global_pool_layers == 'every':
                num_gates = num_layers
            elif global_pool_layers == 'middle':
                num_gates = 1  # Only after layer num_layers//2
            elif global_pool_layers == 'last':
                num_gates = 1  # Only after last conv layer
            else:
                num_gates = num_layers

            # Determine gate input size based on pooling type
            if global_pool_type == 'both':
                gate_input_size = hidden_channels * 3  # node + mean_pool + max_pool
            else:
                gate_input_size = hidden_channels * 2  # node + pool

            self.global_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(gate_input_size, hidden_channels),
                    nn.Sigmoid()
                ) for _ in range(num_gates)
            ])

            # Projection for pooled features if using 'both'
            if global_pool_type == 'both':
                self.global_proj = nn.ModuleList([
                    nn.Linear(hidden_channels * 2, hidden_channels)
                    for _ in range(num_gates)
                ])

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if conv_type == 'gcn':
                conv = GCNConv(hidden_channels, hidden_channels)
            elif conv_type == 'gat':
                conv = GATConv(
                    hidden_channels,
                    hidden_channels // 4,
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            elif conv_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINConv(mlp)
            elif conv_type == 'gine':
                # GIN with Edge features - incorporates edge information
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                conv = GINEConv(mlp, edge_dim=edge_dim)
            elif conv_type == 'gatv2':
                # GATv2 - improved attention mechanism with edge features
                conv = GATv2Conv(
                    hidden_channels,
                    hidden_channels // 4,  # 4 heads × 32 = 128 output
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,  # Important when using edge features
                    residual=True  # Internal skip connection for better gradient flow
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Output layers
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                element_idx=None, residue_idx=None):
        """
        Forward pass.

        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (LongTensor): Edge indices [2, num_edges]
            edge_attr (Tensor, optional): Edge features [num_edges, edge_dim]
            batch (LongTensor, optional): Batch vector [num_nodes]
            element_idx (LongTensor, optional): Element indices for embedding [num_nodes]
            residue_idx (LongTensor, optional): Residue indices for embedding [num_nodes]

        Returns:
            Tensor: Predicted atom exposure values [num_nodes, 1]
        """
        # Apply embeddings if configured
        if self.use_embeddings and element_idx is not None and residue_idx is not None:
            # x contains only numerical features
            element_embed = self.element_embedding(element_idx)
            residue_embed = self.residue_embedding(residue_idx)
            x = torch.cat([x, element_embed, residue_embed], dim=-1)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Create batch vector if not provided (single graph)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Track gate index for global pooling
        gate_idx = 0

        # Graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_in = x

            # Apply convolution
            if self.conv_type == 'gcn':
                x = conv(x, edge_index)
            elif self.conv_type == 'gat':
                x = conv(x, edge_index, edge_attr=edge_attr)
            elif self.conv_type == 'gin':
                x = conv(x, edge_index)
            elif self.conv_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            elif self.conv_type == 'gatv2':
                x = conv(x, edge_index, edge_attr=edge_attr)

            # Batch normalization
            x = bn(x)

            # Activation and Dropout
            # ELU for GATv2 (better gradient flow with attention), ReLU for others
            if self.conv_type == 'gatv2':
                x = F.elu(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            # Add the residual (identity) after the transformation block
            if i > 0:
                x = x + x_in

            # === DYNAMIC GLOBAL POOLING ===
            if self.use_global_pool:
                apply_pooling = False

                if self.global_pool_layers == 'every':
                    apply_pooling = True
                elif self.global_pool_layers == 'middle' and i == self.num_layers // 2:
                    apply_pooling = True
                elif self.global_pool_layers == 'last' and i == self.num_layers - 1:
                    apply_pooling = True

                if apply_pooling:
                    # 1. Global pooling: aggregate all nodes per graph
                    if self.global_pool_type == 'mean':
                        global_repr = global_mean_pool(x, batch)  # (num_graphs, H)
                    elif self.global_pool_type == 'max':
                        global_repr = global_max_pool(x, batch)  # (num_graphs, H)
                    elif self.global_pool_type == 'both':
                        global_mean = global_mean_pool(x, batch)  # (num_graphs, H)
                        global_max = global_max_pool(x, batch)    # (num_graphs, H)
                        # Project concatenated pooling back to hidden_channels
                        global_repr = self.global_proj[gate_idx](
                            torch.cat([global_mean, global_max], dim=-1)
                        )  # (num_graphs, H)

                    # 2. Broadcast: expand global representation to node level
                    global_expanded = global_repr[batch]  # (num_nodes, H)

                    # 3. Gated combination: learn how much global context to use
                    if self.global_pool_type == 'both':
                        # For 'both', we pass node + mean + max to gate
                        global_mean_expanded = global_mean_pool(x, batch)[batch]
                        global_max_expanded = global_max_pool(x, batch)[batch]
                        gate_input = torch.cat([x, global_mean_expanded, global_max_expanded], dim=-1)
                    else:
                        gate_input = torch.cat([x, global_expanded], dim=-1)

                    gate = self.global_gates[gate_idx](gate_input)  # (num_nodes, H)

                    # 4. Apply gated global context
                    x = x + gate * global_expanded

                    gate_idx += 1

        # Output projection
        out = self.out_proj(x)

        return out.squeeze(-1)


class SimpleGCN(nn.Module):
    """
    Simple baseline GCN model.

    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden units
        num_layers (int): Number of layers
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.out = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out(x)
        return x.squeeze(-1)


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
        config (dict): Model configuration

    Returns:
        nn.Module: Initialized model
    """
    model_type = config.get('model_type', 'gnn')

    if model_type == 'gnn':
        return AtomExposureGNN(
            in_channels=config['in_channels'],
            hidden_channels=config.get('hidden_channels', 128),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.2),
            conv_type=config.get('conv_type', 'gcn'),
            edge_dim=config.get('edge_dim', 1),
            use_embeddings=config.get('use_embeddings', False),
            num_numerical=config.get('num_numerical', 31),
            num_elements=config.get('num_elements', 5),
            num_residues=config.get('num_residues', 21),
            element_embed_dim=config.get('element_embed_dim', 8),
            residue_embed_dim=config.get('residue_embed_dim', 11),
            use_global_pool=config.get('use_global_pool', False),
            global_pool_type=config.get('global_pool_type', 'mean'),
            global_pool_layers=config.get('global_pool_layers', 'every')
        )
    elif model_type == 'simple':
        return SimpleGCN(
            in_channels=config['in_channels'],
            hidden_channels=config.get('hidden_channels', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'minimal':
        return MinimalGCN(
            in_channels=config['in_channels'],
            hidden_channels=config.get('hidden_channels', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            use_global_pool=config.get('use_global_pool', False),
            global_pool_type=config.get('global_pool_type', 'mean'),
            feature_noise=config.get('feature_noise', 0.0)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    # Test model instantiation
    model = AtomExposureGNN(in_channels=80, hidden_channels=128, num_layers=3)
    print(model)

    # Test forward pass
    x = torch.randn(100, 80)  # 100 nodes, 80 features
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges

    out = model(x, edge_index)
    print(f"\nOutput shape: {out.shape}")
    print(f"Output range: [{out.min().item():.2f}, {out.max().item():.2f}]")
