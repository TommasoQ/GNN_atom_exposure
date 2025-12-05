# Installation Guide

This guide provides detailed installation instructions for the GNN Protein Atom Exposure Prediction project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

## Installation Methods

### Method 1: Automated Installation (Recommended)

We provide installation scripts that handle the proper installation order for PyTorch Geometric dependencies.

#### For CPU-only (Simple and Fast)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run simple installation script
bash install_simple.sh
```

#### For Automatic GPU/CPU Detection

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run full installation script
bash install.sh
```

### Method 2: Manual Installation

If you prefer to install packages manually or the scripts don't work on your system:

#### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install PyTorch

**For CPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check [PyTorch website](https://pytorch.org/get-started/locally/) for other CUDA versions.

#### Step 3: Install PyTorch Geometric

```bash
pip install torch-geometric
```

#### Step 4: Install PyG Extension Packages

**For CPU:**
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**For GPU (CUDA 11.8):**
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**For GPU (CUDA 12.1):**
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

#### Step 5: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

After installation, verify everything is working:

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test CUDA availability (if GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test PyTorch Geometric
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

# Test all imports
python -c "
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import pandas
import numpy
import matplotlib
print('All imports successful!')
"
```

## Troubleshooting

### Issue: torch-scatter installation fails

**Solution:** Install PyTorch first before installing torch-scatter. The extension packages require PyTorch to be present.

```bash
# Install in this order:
pip install torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue: CUDA version mismatch

**Solution:** Ensure PyTorch and PyG extensions use the same CUDA version.

```bash
# Check your CUDA version
nvcc --version

# Install matching versions
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Issue: Graphein installation fails

**Solution:** Graphein has many dependencies. Install them separately:

```bash
pip install biopython networkx pandas numpy
pip install graphein
```

### Issue: Out of memory during installation

**Solution:** Install packages one at a time:

```bash
pip install torch
pip install torch-geometric
pip install pandas numpy
# ... continue with other packages
```

### Issue: Permission errors

**Solution:** Use a virtual environment or install with user flag:

```bash
pip install --user package-name
```

## Platform-Specific Notes

### Linux

Should work out of the box with the provided scripts.

### macOS

- CUDA is not supported on macOS. Use CPU version.
- Some packages might need Xcode command line tools:
  ```bash
  xcode-select --install
  ```

### Windows

- Use Git Bash or WSL2 to run `.sh` scripts
- Or follow the manual installation steps in PowerShell/CMD
- PyTorch with CUDA works on Windows with proper NVIDIA drivers

## Minimal Installation (For Testing)

If you just want to test the code structure without training:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install pandas numpy matplotlib pyyaml tqdm
```

## Development Installation

If you want to modify the code and contribute:

```bash
# Install in development mode
pip install -e .

# Install additional dev dependencies
pip install pytest black flake8 mypy
```

## Docker Installation (Alternative)

Coming soon: Docker container with all dependencies pre-installed.

## Next Steps

After successful installation:

1. Verify dataset structure: `ls dataset/`
2. Run data exploration: `jupyter notebook notebooks/01_data_exploration.ipynb`
3. Train a model: `python main.py --epochs 10`

## Getting Help

- Check PyTorch installation: https://pytorch.org/get-started/locally/
- Check PyG installation: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Open an issue if problems persist
