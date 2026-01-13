# Folder Structure Explanation & Cleanup Guide

## Current Folder Structure

```
GNN_atom_exposure/
├── processed/          # 4,596 files (4.7 GB) - BASE LEVEL ⚠️ DUPLICATE
├── raw/                # Empty - BASE LEVEL ⚠️ DUPLICATE
└── dataset/
    ├── processed/      # 3,677 files (3.8 GB) - DATASET LEVEL ⚠️ DUPLICATE
    ├── raw/            # Empty - DATASET LEVEL ⚠️ DUPLICATE
    └── sadic_data/     # 4,767 protein dirs (19 MB CSV files) ✓ SOURCE DATA
```

---

## What Each Folder Contains

### 1. `sadic_data/` - **SOURCE DATA** ✓ KEEP THIS

**Location**: `dataset/sadic_data/`
**Size**: ~19 MB (CSV files)
**Contents**: Raw protein structure data from SADIC pipeline

```
sadic_data/
├── 142l/
│   ├── 142l__graphein__ATOM_nodes.csv   # 74 features per atom
│   ├── 142l__graphein__ATOM_edges.csv   # Edge list
│   ├── 142l__graphein__pdb_df.csv
│   ├── 142l__graphein__raw_pdb_df.csv
│   └── 142l__graphein__rgroup_df.csv
├── 144l/
├── ... (4,767 proteins total)
```

**Purpose**:
- Contains original CSV files with atom features
- Used by `dataset_fixed.py` to load protein graphs
- **This is the SOURCE - DO NOT DELETE**

---

### 2. `processed/` - **CACHED PYTORCH TENSORS** ⚠️ REDUNDANT

**Locations**:
- `./processed/` (base level) - 4,596 files, 4.7 GB
- `./dataset/processed/` (dataset level) - 3,677 files, 3.8 GB

**Contents**: Pre-processed PyTorch `.pt` files

```
processed/
├── 142l.pt     # 567 KB - cached PyTorch graph
├── 144l.pt     # 575 KB
├── 1a2f.pt     # 1.1 MB
└── ... (thousands more)
```

**Purpose**:
- PyTorch Geometric's automatic caching mechanism
- After first load from CSV, saves processed graph as `.pt`
- Speeds up subsequent loads

**Problem**:
- ❌ **TWO processed folders** exist (base + dataset)
- ❌ Different file counts (4,596 vs 3,677)
- ❌ Created at different times
- ❌ Wasting 8.5 GB total disk space

---

### 3. `raw/` - **EMPTY PLACEHOLDERS** ⚠️ DELETE

**Locations**:
- `./raw/` (base level) - Empty
- `./dataset/raw/` (dataset level) - Empty

**Purpose**:
- PyTorch Geometric convention for "raw unprocessed data"
- Typically would contain downloaded PDB files
- In our case, we use `sadic_data/` instead
- **Both are EMPTY and unused**

---

## Why This Happened

### PyTorch Geometric Dataset Behavior:

When you initialize a PyTorch Geometric Dataset:
```python
dataset = ProteinAtomDataset(root=".")
```

It automatically creates:
```
root/
├── raw/        # For raw data downloads
└── processed/  # For cached processed graphs
```

### Our Project's Confusion:

1. **First attempt**: `root="."` (project base)
   - Created: `./processed/` and `./raw/`

2. **Second attempt**: `root="./dataset"`
   - Created: `./dataset/processed/` and `./dataset/raw/`

3. **Current**: `root="."` but dataset code adds `dataset/`
   - Now correctly uses `./dataset/sadic_data/`
   - But old cached files remain!

---

## The Difference Between Folders

### `sadic_data/` vs `processed/`:

| Feature | sadic_data/ | processed/ |
|---------|-------------|------------|
| **Format** | CSV files | PyTorch .pt files |
| **Size** | 19 MB | 8.5 GB total |
| **Content** | Raw features + edges | Cached torch tensors |
| **Purpose** | Source data | Speed optimization |
| **Needed?** | ✅ YES - SOURCE | ⚠️ Cache (regenerated) |
| **Human readable?** | ✅ Yes (CSV) | ❌ No (binary) |

### Why both exist:

```
Load process:
1. dataset_fixed.py reads sadic_data/142l/*.csv
2. Processes into torch tensors
3. Saves to processed/142l.pt for faster future loading
4. Next time: just loads processed/142l.pt directly
```

---

## Recommended Cleanup

### ⚠️ Safe to Delete (Will be regenerated):

```bash
# Delete duplicate processed folders
rm -rf ./processed/              # 4.7 GB freed
rm -rf ./dataset/processed/      # 3.8 GB freed
```

**Result**: Saves 8.5 GB disk space
**Impact**: None! Will regenerate on next run

### ⚠️ Safe to Delete (Empty):

```bash
# Delete empty raw folders
rm -rf ./raw/
rm -rf ./dataset/raw/
```

**Result**: Cleaner structure
**Impact**: None (they're empty)

### ✅ KEEP (SOURCE DATA):

```bash
# DO NOT DELETE:
dataset/sadic_data/              # 19 MB - THIS IS YOUR DATA
dataset/depth_indexes.pkl        # Labels
dataset/depth_indexes_dict.pkl   # Optimized labels
dataset/protein_sample_5000.csv  # Protein list
```

---

## Clean Folder Structure (Recommended)

After cleanup:

```
GNN_atom_exposure/
└── dataset/
    ├── sadic_data/              # ✓ Source CSV files (19 MB)
    ├── depth_indexes.pkl        # ✓ Labels (330 MB)
    ├── depth_indexes_dict.pkl   # ✓ Optimized labels (fast load)
    └── protein_sample_5000.csv  # ✓ Protein list (53 KB)
```

**Note**: `processed/` will be recreated automatically when you run training, but only with the files actually needed (should be ~3.8 GB for 3,675 training proteins).

---

## Should You Delete `processed/`?

### Pros of Deleting:
- ✅ Saves 8.5 GB disk space
- ✅ Removes stale/duplicate caches
- ✅ Cleaner project structure
- ✅ Forces regeneration with current code

### Cons of Deleting:
- ⏱️ First training run will be slower (regenerates cache)
- ⏱️ Takes ~2-3 minutes to process all proteins once

### Recommendation:

**DELETE IT** if:
- You need disk space
- You want a clean slate
- You've changed the dataset code

**KEEP IT** if:
- You're actively training (saves time)
- Disk space isn't an issue
- Cache is up-to-date with current code

---

## Implementation Commands

### Option 1: Full Cleanup (Recommended)

```bash
# Navigate to project
cd C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure

# Delete duplicate caches and empty folders
rm -rf ./processed/
rm -rf ./dataset/processed/
rm -rf ./raw/
rm -rf ./dataset/raw/

echo "Cleaned up 8.5 GB! processed/ will regenerate on next training run."
```

### Option 2: Keep One Cache (Conservative)

```bash
# Keep the dataset/processed/ cache, delete base duplicate
rm -rf ./processed/
rm -rf ./raw/
rm -rf ./dataset/raw/

echo "Cleaned up 4.7 GB, kept dataset cache."
```

### Option 3: Keep Everything (No Cleanup)

```bash
# Do nothing
# Wastes 8.5 GB but training is faster
```

---

## What Happens After Cleanup

### First Training Run After Cleanup:

```bash
python main.py --epochs 50
```

**Output**:
```
Loading datasets...
Loading pre-converted depth_indexes dict...
  Loaded 4594 proteins
Processing protein 142l...
Processing protein 144l...
...
[Takes 2-3 minutes to process all proteins]
Saved processed data to dataset/processed/142l.pt
...
```

### Subsequent Runs:

```bash
python main.py --epochs 50
```

**Output**:
```
Loading datasets...
Loading pre-converted depth_indexes dict...
  Loaded 4594 proteins
Loading from cache: dataset/processed/142l.pt
...
[Much faster!]
```

---

## Summary

| Folder | Purpose | Size | Keep? | Reason |
|--------|---------|------|-------|--------|
| `dataset/sadic_data/` | Source CSV data | 19 MB | ✅ YES | Original data |
| `dataset/processed/` | PyTorch cache | 3.8 GB | ⚠️ Optional | Regenerated |
| `./processed/` | Old cache | 4.7 GB | ❌ DELETE | Duplicate |
| `dataset/raw/` | Empty | 0 | ❌ DELETE | Unused |
| `./raw/` | Empty | 0 | ❌ DELETE | Unused |

**Recommended Action**: Delete `./processed/`, `./raw/`, `./dataset/raw/`
**Disk Space Saved**: ~4.7 GB
**Risk**: None (regenerates automatically)

---

**Created**: 2026-01-08
**Last Updated**: 2026-01-08
