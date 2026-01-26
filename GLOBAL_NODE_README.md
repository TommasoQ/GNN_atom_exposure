# Global Node Feature for Protein Graphs

## Overview

Il **Virtual Global Node** è una tecnica per aggiungere contesto globale ai protein graphs, permettendo alla rete di:
1. **Imparare proprietà globali** della proteina
2. **Calibrare le predizioni** basandosi sul contesto globale
3. **Ridurre outliers** dove predizioni locali sono sbagliate per mancanza di contesto

## Problema Motivante

### Osservazione in Phase 15 noCC (R² = 0.57)
```
Target ≈ 0.0 (buried)  →  Predizione ≈ 1.5 (exposed)

Problema: La rete non ha contesto globale per capire che:
- Questa proteina ha distribuzione di exposure bassa
- Atomi circostanti sono tutti buried
- La predizione 1.5 è fuori scala per questa proteina specifica
```

**Root cause**: Ogni atomo viene predetto **indipendentemente** senza sapere:
- Quanti atomi ha la proteina
- Qual è la distribuzione generale di exposure
- Se la proteina è generalmente buried o exposed

## Soluzione: Virtual Global Node

### Architettura

```
Proteina con N atomi:

Before Global Node:
┌──────────────────┐
│  Atom 1 ←→ Atom 2│
│    ↑  ╲    ╱  ↑  │
│    │   ╲  ╱   │  │
│  Atom 3 ←→ Atom 4│
└──────────────────┘
Problema: Comunicazione locale, no contesto globale


After Global Node:
┌──────────────────────────┐
│     ╔════════╗            │
│     ║ GLOBAL ║            │
│     ║  NODE  ║            │
│     ╚════════╝            │
│    ↙↓↓↓↓↓↓↓↓↓↓↘           │
│  Atom1 ← → Atom2          │
│    ↑  ╲    ╱  ↑           │
│    │   ╲  ╱   │           │
│  Atom3 ← → Atom4          │
└──────────────────────────┘
Soluzione: Tutti gli atomi comunicano col global node
```

### Implementazione Tecnica

**1. Nodo Virtuale Aggiunto**
- **Index**: N (ultimo nodo, dopo gli N atomi)
- **Features**: Aggregati statistici della proteina
- **Target**: None (mascherato durante training)

**2. Edges Aggiunte** (2N edges totali)
- `Global → Atom i` per tutti i=0...N-1
- `Atom i → Global` per tutti i=0...N-1
- Tipo: Bidirezionale, marcato con flag speciale

**3. Edge Features per Global Edges**
- Features 0-10: Zero (no bond type/distance)
- Feature 11: `1.0` (flag "global edge")

## Global Node Features

Le features del global node sono **aggregati statistici** degli atomi:

### Categoria 1: Numerical Features (24 features)
```python
# Mean e std di tutte le 24 numerical features
mean_b_factor = atoms['b_factor'].mean()
std_b_factor = atoms['b_factor'].std()
mean_hbond_donors = atoms['hbond_donors'].mean()
# ... per tutte le 24 features
```

### Categoria 2: Geometric Features (8 features)
```python
# Mean e std delle 7 geometric features
mean_radial_position = atoms['geom_radial_position'].mean()
std_radial_position = atoms['geom_radial_position'].std()
# ... etc
```

### Categoria 3: Protein Size (2 features)
```python
num_atoms = float(len(atoms))
avg_degree = num_edges / num_atoms
```

### Categoria 4: Target Statistics (4 features, optional)
```python
# Solo durante training, aiuta il modello a imparare la scala
target_mean = atoms['exposure'].mean()
target_std = atoms['exposure'].std()
target_min = atoms['exposure'].min()
target_max = atoms['exposure'].max()
```

**Dimensione totale**: Varia, viene adattata a 92 (o 93) per matchare atom features

## Come Funziona in Pratica

### Forward Pass

1. **Layer 1**: Atomi mandano info iniziale al global node
   ```
   Atom features → GATv2 aggregation → Global node receives context
   ```

2. **Layer 2**: Global node "comprende" il contesto proteina
   ```
   Global node: "Questa proteina ha mean_exposure=0.3, std=0.2"
   ```

3. **Layer 3-4**: Atomi ricevono contesto globale
   ```
   Global node → Broadcast to atoms → "Calibrate your predictions!"
   ```

4. **Output**: Predizioni informate da contesto globale
   ```
   Atom prediction = f(local features, global context)
   ```

### Training

**Mascheramento del Global Node**:
- Global node NON ha target di predizione
- Durante loss calculation: `loss = criterion(pred[mask], target[mask])`
- Mask esclude il global node (target = -1)

**Backward Pass**:
- Gradient flow normale attraverso il global node
- Global node impara features utili per calibrazione

## Benefici Attesi

### 1. **Calibrazione Globale**
```python
# Prima (senza global node):
if radial_position < 0.3:
    pred = 1.5  # Errore! Troppo alto!

# Dopo (con global node):
if radial_position < 0.3 and global_mean_exposure < 0.5:
    pred = 0.2  # Calibrato sul contesto globale
```

### 2. **Riduzione Outliers**
- Predizioni estreme (es. 1.5 quando target=0) vengono corrette
- Global node fornisce "sanity check" sulla scala

### 3. **Information Flow a Lungo Raggio**
```
Atom A (position=0) ←→ Global Node ←→ Atom Z (position=1000)
```
- Atomi lontani comunicano attraverso il global node
- Utile per proteine grandi

### 4. **Implicit Normalization**
- Modello impara automaticamente a normalizzare
- Non serve normalizzazione manuale per proteina

## Configurazione

### Config YAML
```yaml
features:
  use_global_node: true                    # Enable global node
  global_node_include_target_stats: true   # Include target stats (train only)
```

### Parametri
- **`use_global_node`**: `true`/`false` (default: `false`)
- **`global_node_include_target_stats`**: Include statistiche target nel global node
  - `true`: Training migliore ma rischio leakage
  - `false`: Più "fair" ma potrebbe performare peggio

## Files Modificati

### 1. Transform Implementation
**`src/data/global_node_transform.py`**
- `AddGlobalNode`: Transform class completa
- `AddGlobalNodeSimple`: Versione semplificata

### 2. Training Loop
**`src/training/train.py`**
- Modified `train_epoch()`: Mask global node in loss
- Modified `validate()`: Mask global node in metrics

### 3. Main Script
**`main.py`**
- Load transform based on config
- Apply to train/val/test datasets

### 4. Configuration
**`configs/phase15_noCC_globalnode.yaml`**
- Example config with global node enabled

## Confronto Risultati Attesi

| Configuration | R² | MAE | Notes |
|---------------|-----|-----|-------|
| **Phase 15 (93 feat + contact_count)** | 0.8892 | 0.087 | Baseline best |
| **Phase 15 noCC (92 feat)** | 0.5689 | 0.179 | Missing critical feature |
| **Phase 15 noCC + Global Node** | **0.60-0.70?** | **0.15-0.17?** | Hypothesis: partial recovery |

**Aspettativa realistica**:
- Global node **NON sostituirà** contact_count completamente
- Ma potrebbe **ridurre il gap** da R²=0.57 a R²≈0.65-0.70
- Principale beneficio: **riduzione outliers estremi**

## Usage

### Training
```bash
cd GNN_atom_exposure
.venv\Scripts\python.exe main.py --config configs/phase15_noCC_globalnode.yaml
```

### Custom Config
```yaml
features:
  use_global_node: true
  global_node_include_target_stats: true  # Or false for stricter evaluation
```

## Debugging

### Verify Global Node is Added
```python
from src.data.global_node_transform import AddGlobalNode
from src.data.dataset_fixed import ProteinAtomDataset

transform = AddGlobalNode()
dataset = ProteinAtomDataset(root='dataset', split='train', transform=transform)

data = dataset[0]
print(f"Num nodes: {data.num_nodes}")  # Should be N+1
print(f"Global mask: {data.global_node_mask.sum()}")  # Should be 1
print(f"Target shape: {data.y.shape}")  # Should be (N+1,)
print(f"Last target: {data.y[-1]}")  # Should be -1.0
```

### Check Global Edges
```python
global_idx = data.num_nodes - 1
global_edges = (data.edge_index[0] == global_idx) | (data.edge_index[1] == global_idx)
print(f"Global edges: {global_edges.sum()}")  # Should be 2N
```

### Verify Masking in Training
```python
# In training loop, this should print filtered shapes:
print(f"Out shape: {out.shape}")  # (N+1,)
print(f"Out filtered shape: {out[mask].shape}")  # (N,)
```

## Troubleshooting

### Error: "size mismatch"
**Cause**: Global node features dimension doesn't match atom features

**Solution**: Check `_compute_global_features()` padding logic

### Error: "Expected 92 features, got 93"
**Cause**: Transform not applied consistently

**Solution**: Ensure transform is passed to ALL datasets (train/val/test)

### Loss explodes
**Cause**: Global node not masked, trying to predict target=-1

**Solution**: Verify masking in `train.py` line ~235 and ~327

### R² doesn't improve
**Possible reasons**:
1. Global node features not informative enough
2. Model needs more capacity (try more layers/hidden)
3. Contact_count is truly irreplaceable

**Solution**: Analyze attention weights on global edges

## Advanced: Custom Global Features

To customize global node features, modify `_compute_global_features()`:

```python
def _compute_global_features(self, data: Data) -> torch.Tensor:
    features = []

    # Your custom aggregations
    features.append(custom_aggregation_1(data))
    features.append(custom_aggregation_2(data))

    return torch.cat(features)
```

## References

**Similar Work**:
- Graph U-Nets: Global pooling nodes
- Graph Transformers: [CLS] token analogy
- MPNN with global node: "Learning to Simulate Complex Physics with Graph Networks" (DeepMind)

**Key Insight**: Global node = learnable aggregation + broadcast mechanism

---

## Next Steps

1. **Lancia Phase 15 noCC + Global Node**
2. **Confronta con Phase 15 noCC**: R² improvement?
3. **Analisi outliers**: Ridotti?
4. **Feature importance**: Global node features importanti?

Se global node migliora significativamente (R²>0.65), considerare:
- Global node + contact_count (93 features)
- Multiple global nodes (hierarchical)
- Attention analysis su global edges

---

**Conclusione**: Global node è un modo elegante per dare contesto globale alla rete. Non sostituirà contact_count completamente, ma dovrebbe migliorare calibrazione e ridurre outliers. 🧬🌍
