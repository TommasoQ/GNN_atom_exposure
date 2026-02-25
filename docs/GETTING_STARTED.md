# Getting Started

Installation and usage guide for the GNN Protein Atom Exposure Prediction project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-enabled GPU (recommended, not required)

## Installation

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install PyTorch

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

### Step 3: Install PyTorch Geometric

```bash
pip install torch-geometric
```

### Step 4: Install PyG Extension Packages

**For CPU:**
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**For GPU (match your CUDA version):**
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 5: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'PyG: {torch_geometric.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('All imports successful!')
"
```

## Dataset Setup

Ensure the dataset is in place:

```bash
ls dataset/
# Should contain: depth_indexes.pkl, protein_sample_5000.csv, sadic_data/
```

See [Dataset Documentation](DATASET.md) for how to obtain the dataset.

## Training

### Train the best model

```bash
python main.py --config configs/phase15_globalpool.yaml
```

### Train with Gaussian noise regularization

```bash
python main.py --config configs/phase15_globalpool_gaussian.yaml
```

### Custom parameters

```bash
python main.py --config configs/phase15_globalpool.yaml --batch-size 16 --epochs 100
```

## Evaluation

### Evaluate and generate plots (no training)

```bash
python main.py --config configs/phase15_globalpool.yaml --eval-only
```

### Evaluate with a specific checkpoint

```bash
python main.py --config configs/phase15_globalpool.yaml --eval-only --checkpoint path/to/best_model.pt
```

### Force CPU evaluation

```bash
python main.py --config configs/phase15_globalpool.yaml --eval-only --cpu
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config PATH` | Path to YAML config file |
| `--batch-size INT` | Override batch size |
| `--epochs INT` | Override number of epochs |
| `--lr FLOAT` | Override learning rate |
| `--eval-only` | Skip training, only evaluate |
| `--checkpoint PATH` | Load model from checkpoint |
| `--cpu` | Force CPU usage |

## Output Files

After training/evaluation, results are saved to the experiment log directory:

- `training_curves.png` - Loss and R² over epochs
- `predictions_raw.png` - 2D heatmap of raw predictions vs actual
- `predictions_clamped.png` - 2D heatmap of clamped predictions vs actual
- `error_distribution.png` - Error histogram and box plot
- `error_by_exposure_range.png` - MAE and bias per exposure range
- `test_metrics.json` - All numerical metrics

## Troubleshooting

### Out of Memory
Reduce batch size: `--batch-size 8` or `--batch-size 4`

### Slow Training
- Use GPU if available
- Enable AMP (already enabled in default configs)
- Reduce `num_workers` in config

### Import Errors
Ensure you're in the project root with the virtual environment activated:
```bash
cd GNN_atom_exposure
source venv/bin/activate
```

### torch-scatter installation fails
Install PyTorch first, then torch-scatter with the matching CUDA version URL.

## See Also

- [Architecture](ARCHITECTURE.md) - Model design
- [Dataset](DATASET.md) - Data structure
- [History](HISTORY.md) - Experimental timeline
