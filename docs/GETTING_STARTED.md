# Getting Started

This guide will help you install and start using the GNN Protein Atom Exposure Prediction project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

## Installation

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

### Verify Installation

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

## Quick Start

### Verify Dataset

Check that your dataset is properly structured:

```bash
ls -lh dataset/
# Should show: depth_indexes.pkl, protein_sample_5000.csv, sadic_data/
```

### Train a Model

Train a model with default configuration:

```bash
python main.py
```

Train with custom parameters:

```bash
python main.py --batch-size 16 --epochs 50 --lr 0.0005
```

Evaluate a trained model:

```bash
python main.py --eval-only --checkpoint experiments/checkpoints/best_model.pt --visualize
```

### Configuration

Edit `configs/config.yaml` to customize:

- **Data settings**: batch size, splits
- **Model architecture**: hidden dimensions, number of layers, GNN type
- **Training parameters**: learning rate, epochs, weight decay
- **Experiment settings**: checkpoint directory, logging

### Command Line Arguments

```bash
python main.py --help
```

Available options:
- `--config PATH`: Path to config file (default: configs/config.yaml)
- `--batch-size INT`: Batch size
- `--epochs INT`: Number of training epochs
- `--lr FLOAT`: Learning rate
- `--eval-only`: Skip training, only evaluate
- `--checkpoint PATH`: Load model from checkpoint
- `--visualize`: Generate result visualizations
- `--cpu`: Force CPU usage (even if GPU available)

## Using Python API

Create a custom training script:

```python
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from src.training.train import train_model
from torch_geometric.loader import DataLoader

# Load data
train_dataset = ProteinAtomDataset(root='dataset/', split='train')
val_dataset = ProteinAtomDataset(root='dataset/', split='val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Create model
model = AtomExposureGNN(
    in_channels=88,  # Current feature count
    hidden_channels=128,
    num_layers=3,
    dropout=0.2
)

# Train
trainer = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.001
)
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python main.py --batch-size 4
```

### Slow Training

- Use GPU if available
- Reduce number of workers in config
- Use smaller model (fewer layers/channels)

### Import Errors

Make sure you're in the project root and virtual environment is activated:
```bash
cd GNN_atom_exposure
source venv/bin/activate
```

### torch-scatter installation fails

**Solution:** Install PyTorch first before installing torch-scatter. The extension packages require PyTorch to be present.

```bash
# Install in this order:
pip install torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### CUDA version mismatch

**Solution:** Ensure PyTorch and PyG extensions use the same CUDA version.

```bash
# Check your CUDA version
nvcc --version

# Install matching versions
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Graphein installation fails

**Solution:** Graphein has many dependencies. Install them separately:

```bash
pip install biopython networkx pandas numpy
pip install graphein
```

### Permission errors

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

## Next Steps

1. Read the [Architecture documentation](ARCHITECTURE.md) to understand the model
2. Review the [Dataset documentation](DATASET.md) to understand the data
3. Check [Experiments documentation](EXPERIMENTS.md) for current results
4. Explore the data with Jupyter notebooks
5. Train your first model!

## Resources

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graphein: https://github.com/a-r-j/graphein
- Project documentation: See [docs/](.) for full documentation
