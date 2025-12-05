# Quick Start Guide

This guide will help you get started with the GNN Protein Atom Exposure Prediction project.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

**Quick Install (Recommended):**

```bash
bash install_simple.sh
```

**Or see [INSTALL.md](INSTALL.md) for detailed installation instructions.**

### 3. Verify Dataset

Check that your dataset is properly structured:

```bash
ls -lh dataset/
# Should show: depth_indexes.pkl, protein_sample_5000.csv, sadic_data/
```

## Usage

### Option 1: Using the Main Script (Recommended)

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

### Option 2: Using Jupyter Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open and run the notebooks in order:
1. `notebooks/01_data_exploration.ipynb` - Explore and understand the dataset
2. Create your own training notebook based on the examples

### Option 3: Python Script

Create a custom training script:

```python
from src.data.dataset import ProteinAtomDataset
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
    in_channels=80,
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

## Configuration

Edit `configs/config.yaml` to customize:

- **Data settings**: batch size, splits
- **Model architecture**: hidden dimensions, number of layers, GNN type
- **Training parameters**: learning rate, epochs, weight decay
- **Experiment settings**: checkpoint directory, logging

## Command Line Arguments

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

## Project Structure

```
GNN-atoms/
├── main.py                    # Main training script
├── requirements.txt           # Python dependencies
├── configs/                   # Configuration files
├── dataset/                   # Your dataset
├── src/                       # Source code
│   ├── data/                  # Dataset classes
│   ├── models/                # GNN models
│   ├── training/              # Training and evaluation
│   └── utils/                 # Utilities
├── notebooks/                 # Jupyter notebooks
├── experiments/               # Saved models and logs
└── results/                   # Predictions and plots
```

## Common Tasks

### Train a Model

```bash
python main.py --epochs 100
```

### Evaluate on Test Set

```bash
python main.py --eval-only --checkpoint experiments/checkpoints/best_model.pt
```

### Generate Visualizations

```bash
python main.py --eval-only --visualize --checkpoint experiments/checkpoints/best_model.pt
```

### Try Different GNN Architectures

Edit `configs/config.yaml`:

```yaml
model:
  conv_type: gat  # Options: gcn, gat, gin
  hidden_channels: 256
  num_layers: 4
```

### Use Different Train/Val/Test Splits

Edit `src/data/dataset.py` to modify split ratios or implement custom splitting logic.

## Expected Results

After training, you should see:

- Training and validation loss curves
- Model checkpoints in `experiments/checkpoints/`
- Evaluation metrics (MAE, RMSE, R²)
- Prediction visualizations (if `--visualize` is used)

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
cd /home/tquintab/GNN-atoms
source venv/bin/activate
```

### Installation Errors

If you get errors installing torch-scatter or other PyG extensions:
```bash
# Install PyTorch first
pip install torch

# Then install PyG extensions
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Then remaining dependencies
pip install -r requirements.txt
```

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Next Steps

1. Run data exploration notebook to understand the dataset
2. Train a baseline model with default settings
3. Experiment with different architectures (GCN, GAT, GIN)
4. Tune hyperparameters (learning rate, hidden dimensions, etc.)
5. Analyze predictions and errors
6. Implement advanced features (attention, edge features, etc.)

## Resources

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graphein: https://github.com/a-r-j/graphein
- Project README: See README.md for full documentation

## Support

For issues or questions, check the project README or create an issue in the repository.
