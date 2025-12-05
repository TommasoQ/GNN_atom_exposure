#!/bin/bash

# Simple installation script - installs CPU-only version
# This is the most reliable method that works on all systems

set -e

echo "=================================="
echo "Simple Installation (CPU only)"
echo "=================================="

# Step 1: Install PyTorch
echo ""
echo "Step 1/4: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install PyTorch Geometric
echo ""
echo "Step 2/4: Installing PyTorch Geometric..."
pip install torch-geometric

# Step 3: Install PyG optional extensions (may skip if not available)
echo ""
echo "Step 3/4: Installing PyG extensions (optional)..."
# Try to install extensions, but don't fail if they're not available
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html || echo "Note: Some PyG extensions not available, continuing anyway..."

# Step 4: Install remaining requirements
echo ""
echo "Step 4/4: Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "Verifying installation..."
python -c "import torch; import torch_geometric; print('✓ PyTorch and PyTorch Geometric installed successfully!')" || echo "⚠ Warning: Some imports failed"
echo ""
echo "You can now run:"
echo "  python main.py --help"
echo ""
