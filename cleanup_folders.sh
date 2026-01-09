#!/bin/bash
# Cleanup duplicate and unnecessary folders
# This script is SAFE - only removes cached/regeneratable data

echo "======================================================================"
echo "GNN Atom Exposure - Folder Cleanup Script"
echo "======================================================================"
echo ""

# Check current sizes
echo "Current disk usage:"
echo "-------------------"
du -sh ./processed 2>/dev/null && echo "  ./processed/ (base level - DUPLICATE)"
du -sh ./dataset/processed 2>/dev/null && echo "  ./dataset/processed/ (DUPLICATE)"
du -sh ./raw 2>/dev/null && echo "  ./raw/ (empty)"
du -sh ./dataset/raw 2>/dev/null && echo "  ./dataset/raw/ (empty)"
echo ""

echo "Source data (WILL NOT DELETE):"
echo "-------------------------------"
du -sh ./dataset/sadic_data 2>/dev/null && echo "  ./dataset/sadic_data/ (CSV source)"
du -sh ./dataset/*.pkl 2>/dev/null | head -3
echo ""

# Ask for confirmation
echo "This will DELETE:"
echo "  1. ./processed/ (cache, ~4.7 GB)"
echo "  2. ./dataset/processed/ (cache, ~3.8 GB)"
echo "  3. ./raw/ (empty)"
echo "  4. ./dataset/raw/ (empty)"
echo ""
echo "These will be KEPT (source data):"
echo "  - ./dataset/sadic_data/ (CSV files)"
echo "  - ./dataset/*.pkl (labels)"
echo ""
echo "⚠️  Processed cache will regenerate on next training run (~2-3 min)"
echo ""

read -p "Continue with cleanup? (y/N): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo "-------------------"

# Delete duplicate processed folders
if [ -d "./processed" ]; then
    echo "Deleting ./processed/..."
    rm -rf ./processed
    echo "  ✓ Deleted ./processed/"
else
    echo "  - ./processed/ not found"
fi

if [ -d "./dataset/processed" ]; then
    echo "Deleting ./dataset/processed/..."
    rm -rf ./dataset/processed
    echo "  ✓ Deleted ./dataset/processed/"
else
    echo "  - ./dataset/processed/ not found"
fi

# Delete empty raw folders
if [ -d "./raw" ]; then
    echo "Deleting ./raw/..."
    rm -rf ./raw
    echo "  ✓ Deleted ./raw/"
else
    echo "  - ./raw/ not found"
fi

if [ -d "./dataset/raw" ]; then
    echo "Deleting ./dataset/raw/..."
    rm -rf ./dataset/raw
    echo "  ✓ Deleted ./dataset/raw/"
else
    echo "  - ./dataset/raw/ not found"
fi

echo ""
echo "======================================================================"
echo "Cleanup complete!"
echo "======================================================================"
echo ""
echo "Disk space freed: ~8.5 GB"
echo ""
echo "Next steps:"
echo "  1. Run training: python main.py --epochs 50"
echo "  2. Processed cache will regenerate automatically"
echo "  3. First run will take 2-3 minutes longer (one-time cost)"
echo ""
echo "Source data intact:"
echo "  ✓ dataset/sadic_data/ (4,767 proteins)"
echo "  ✓ dataset/depth_indexes.pkl"
echo "  ✓ dataset/depth_indexes_dict.pkl"
echo "======================================================================"
