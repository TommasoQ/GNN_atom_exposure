#!/bin/bash

# Installation script for GNN Protein Atom Exposure Prediction
# This script handles the proper installation order for PyTorch Geometric dependencies

set -e  # Exit on error

echo "=================================="
echo "GNN Project Installation Script"
echo "=================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Install PyTorch
echo ""
echo "Step 1/3: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install PyTorch Geometric and extensions
echo ""
echo "Step 2/3: Installing PyTorch Geometric..."

# Get PyTorch and CUDA versions
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu')")

echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"

# Install PyG extensions from wheel
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Step 3: Install remaining dependencies
echo ""
echo "Step 3/3: Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; import torch_geometric; print(\"Success!\")'"
echo ""
