"""Debug script to check batch handling"""
import sys
sys.path.insert(0, 'src')

from data.dataset_fixed import ProteinAtomDataset
from torch_geometric.loader import DataLoader

feature_config = {'use_aggregated': True}
dataset = ProteinAtomDataset(root='dataset', split='train', feature_config=feature_config)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

batch = next(iter(loader))
print(f"Batch x shape: {batch.x.shape}")
print(f"Has element_idx: {hasattr(batch, 'element_idx')}")
print(f"Has residue_idx: {hasattr(batch, 'residue_idx')}")

if hasattr(batch, 'element_idx'):
    print(f"element_idx shape: {batch.element_idx.shape}")
    print(f"element_idx dtype: {batch.element_idx.dtype}")
    print(f"element_idx unique: {batch.element_idx.unique().tolist()}")

if hasattr(batch, 'residue_idx'):
    print(f"residue_idx shape: {batch.residue_idx.shape}")
    print(f"residue_idx dtype: {batch.residue_idx.dtype}")

# Check what happens with getattr
elem = getattr(batch, 'element_idx', None)
res = getattr(batch, 'residue_idx', None)
print(f"\ngetattr element_idx: {elem is not None}")
print(f"getattr residue_idx: {res is not None}")

# Test model forward
from models.gnn import AtomExposureGNN
import torch

model = AtomExposureGNN(
    in_channels=50,
    hidden_channels=128,
    num_layers=4,
    dropout=0.1,
    conv_type='gatv2',
    edge_dim=11,
    use_embeddings=True,
    num_numerical=31,
    num_elements=5,
    num_residues=21,
    element_embed_dim=8,
    residue_embed_dim=11
)

model.eval()
with torch.no_grad():
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                element_idx=elem, residue_idx=res)
    print(f"\nModel output shape: {out.shape}")
    print(f"Model output range: {out.min().item():.3f} to {out.max().item():.3f}")
    print(f"Target range: {batch.y.min().item():.3f} to {batch.y.max().item():.3f}")
