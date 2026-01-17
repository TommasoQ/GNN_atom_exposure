"""
Graph Neural Network Models for Atom Exposure Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GINEConv, global_mean_pool
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
        residue_embed_dim: int = 11
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim

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
                    hidden_channels // 4,
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False  # Important when using edge features
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            # Add the residual (identity) after the transformation block
            if i > 0:
                x = x + x_in

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
            residue_embed_dim=config.get('residue_embed_dim', 11)
        )
    elif model_type == 'simple':
        return SimpleGCN(
            in_channels=config['in_channels'],
            hidden_channels=config.get('hidden_channels', 64),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
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
