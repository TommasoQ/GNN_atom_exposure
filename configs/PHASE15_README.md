# Phase 15: Optimized Model Configuration

## Overview
Phase 15 è una configurazione **ottimizzata** basata sull'analisi approfondita dei risultati di Phase 14. L'obiettivo è raggiungere performance superiori con maggiore stabilità e migliore generalizzazione.

## Analisi del Problema (da Phase 14)

### Risultati Phase 14
- **R² Test**: 0.8888 (+0.8% vs Phase 13a: 0.8817)
- **Parametri**: 513K (7.2x vs Phase 13a: 71K)
- **ROI**: Solo +0.8% performance con 7x parametri → **Rendimento molto basso**

### Problemi Identificati

#### 1. Training Instability - Spike Anomali
Durante il training di Phase 14, si sono verificati **spike di validation loss** molto grandi:
- Epoch 33-34, 48, 55-57, 67, 70, 73, 75, 80, 82
- Val loss salta da ~0.018-0.020 a **0.048-0.049** (2.5x)
- Causa: Learning rate troppo alto (0.0009) + gradient clipping insufficiente (1.0)

#### 2. Systematic Bias Non Risolto
Il modello continua ad avere **bias direzionale** per range di esposizione:

| Range | MAE | **Bias** | Problema |
|-------|-----|----------|----------|
| Buried (0-0.2) | 0.0718 | **+0.0565** | Forte sovrastima |
| Semi-buried (0.2-0.5) | 0.0879 | +0.0121 | Leggera sovrastima |
| Intermediate (0.5-0.8) | 0.0932 | -0.0240 | Leggera sottostima |
| Semi-exposed (0.8-1.2) | 0.0869 | -0.0478 | Sottostima moderata |
| Exposed (1.2+) | 0.0905 | **-0.0803** | Forte sottostima |

**Pattern sistematico**: Il modello "regredisce alla media"
- Sovrastima atomi buried (poco esposti)
- Sottostima atomi exposed (molto esposti)

#### 3. Modello Troppo Grande per il Dataset
- 513K parametri sono **eccessivi** per ~4500 proteine
- Rischio overfitting nonostante dropout e regularization
- Convergenza lenta (200 epoche)

---

## Strategia Phase 15

### 1. Riduzione Capacità Modello (~50%)
**Obiettivo**: Modello più compatto (~250K params) per migliore fit al dataset

| Parametro | Phase 14 | Phase 15 | Δ |
|-----------|----------|----------|---|
| **num_layers** | 5 | 4 | -20% |
| **hidden_channels** | 176 | 136 | -23% |
| **Parametri totali** | 513K | **254K** | **-50%** |

**Motivazione**:
- Dataset size: ~4500 proteine, ~1M atomi
- 254K params → rapporto ~4 atomi/parametro (più ragionevole)
- 4 layers = sufficiente depth per GNN proteiche

### 2. Training Più Conservativo e Stabile

#### Learning Rate Reduction
```yaml
max_lr: 0.0007  # da 0.0009 (-22%)
```
- **Problema**: Spike di val_loss a 0.0009
- **Soluzione**: LR più basso → gradient updates più stabili

#### Tighter Gradient Clipping
```yaml
gradient_clip: 0.5  # da 1.0 (-50%)
```
- **Problema**: Alcuni batch causano gradient explosions
- **Soluzione**: Clipping più aggressivo previene spike

#### Longer Warmup
```yaml
warmup_epochs: 50  # da 40 (+25%)
```
- **Problema**: Warmup troppo veloce destabilizza early training
- **Soluzione**: Warmup più graduale → start più smooth

#### Regularization Adjustment
```yaml
dropout: 0.26        # da 0.28 (-7%, modello più piccolo)
weight_decay: 1.2e-4 # da 1.5e-4 (-20%, modello più piccolo)
batch_size: 32       # da 28 (+14%, modello più piccolo consuma meno memoria)
```

### 3. **NOVITÀ**: Range-Specific Weighted Loss

#### Problema con la Loss Attuale
La `ExposureWeightedMSELoss` di Phase 14:
- Da più peso ad atomi **ad alta esposizione** (>threshold)
- Ma questo **non risolve il bias direzionale**:
  - Buried vengono comunque sovrastimati
  - Exposed vengono comunque sottostimati

#### Soluzione: `RangeSpecificWeightedMSELoss`

**Idea**: Applicare pesi diversi per **range di esposizione** + **penalizzare errori direzionali**

```python
class RangeSpecificWeightedMSELoss:
    # Base weights per range
    buried (0-0.2):        weight = 1.5x
    semi_buried (0.2-0.5): weight = 1.0x
    intermediate (0.5-0.8): weight = 1.0x
    semi_exposed (0.8-1.2): weight = 1.3x
    exposed (1.2+):        weight = 2.0x

    # Asymmetric penalty (penalty direzionale)
    if (buried) and (pred > target):  # Sovrastima buried
        weight *= 1.5x  # Penalty extra
    if (exposed) and (pred < target):  # Sottostima exposed
        weight *= 1.5x  # Penalty extra
```

**Esempio**:
- Atomo buried: target=0.15, pred=0.20 (sovrastima di +0.05)
  - Base weight: 1.5x (buried)
  - Asymmetric penalty: 1.5x (pred > target)
  - **Total weight: 1.5 × 1.5 = 2.25x** (forte penalizzazione)

- Atomo exposed: target=1.40, pred=1.20 (sottostima di -0.20)
  - Base weight: 2.0x (exposed)
  - Asymmetric penalty: 1.5x (pred < target)
  - **Total weight: 2.0 × 1.5 = 3.0x** (penalizzazione molto forte)

**Effetto Atteso**: Il modello impara a:
- Non sovrastimare atomi buried
- Non sottostimare atomi exposed
- Ridurre il bias sistematico

---

## Configurazione Completa

### Model Architecture
```yaml
model:
  in_channels: 93
  hidden_channels: 136      # 4 heads × 34 channels/head
  num_layers: 4
  dropout: 0.26
  conv_type: gatv2
  edge_dim: 12

Total Parameters: 253,913 (~254K)
```

### Training Hyperparameters
```yaml
training:
  # Optimizer
  learning_rate: 5.0e-04
  weight_decay: 1.2e-04
  gradient_clip: 0.5

  # Scheduler (OneCycleLR)
  scheduler: one_cycle
  max_lr: 0.0007
  warmup_epochs: 50
  div_factor: 25.0
  final_div_factor: 10000.0

  # Data
  batch_size: 32
  num_epochs: 200

  # Early stopping
  early_stopping_patience: 60
  early_stopping_metric: r2

  # Loss function
  weighted_loss: true
  loss_type: range_specific    # NEW!
  loss_range_weights:
    buried: 1.5
    semi_buried: 1.0
    intermediate: 1.0
    semi_exposed: 1.3
    exposed: 2.0
```

---

## Come Lanciare il Training

### 1. Verifica Configurazione (Opzionale)
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe -c "from src.utils.config import Config; cfg = Config.from_yaml('configs/phase15_optimized.yaml'); print('Config OK')"
```

### 2. Lancia Training
```bash
.venv/Scripts/python.exe main.py --config configs/phase15_optimized.yaml
```

### 3. Output Atteso
```
Creating model...
  Architecture: GATV2
  Layers: 4, Hidden: 136
  Parameters: 253,913
  Loss: Range-Specific Weighted MSE
    - Buried (0-0.2): 1.5x, Semi-buried (0.2-0.5): 1.0x
    - Intermediate (0.5-0.8): 1.0x, Semi-exposed (0.8-1.2): 1.3x
    - Exposed (1.2+): 2.0x, Asymmetric penalty: 1.5x
  Scheduler: One Cycle (will initialize after data loading)
```

---

## Aspettative e Metriche di Successo

### Performance Target
| Metrica | Phase 13a | Phase 14 | **Phase 15 Target** |
|---------|-----------|----------|---------------------|
| **R² Test** | 0.8817 | 0.8888 | **0.895+** |
| **Parametri** | 71K | 513K | 254K |
| **ROI** | Baseline | +0.8% | **+1.5-2.0%** |

### Bias Reduction Target
| Range | Phase 14 Bias | **Phase 15 Target** |
|-------|---------------|---------------------|
| Buried (0-0.2) | +0.0565 | **< +0.02** |
| Exposed (1.2+) | -0.0803 | **< -0.03** |

### Training Stability
- **NO spike di validation loss** (>0.03)
- Convergenza smooth senza oscillazioni
- Early stopping tra epoch 120-180

---

## Monitoraggio Durante il Training

### Metriche Chiave da Osservare

#### 1. Validation Loss Stability
```bash
# Check per spike
grep "val_loss" experiments/logs/training_history.csv | awk -F',' '$3 > 0.03 {print "Spike at epoch", $1, ":", $3}'
```
**Successo**: Nessun spike > 0.03

#### 2. Bias per Range (End of Training)
Nell'output finale, controllare:
```
Error by Exposure Range:
Buried (0-0.2):      Bias: +0.0XXX  ← Deve essere < +0.02
Exposed (1.2+):      Bias: -0.0XXX  ← Deve essere < -0.03
```

#### 3. Training vs Validation Gap
```
Final epoch:
train_loss: ~0.015
val_loss:   ~0.017-0.018
Gap: < 20%
```
**Successo**: Gap moderato (no overfitting)

---

## Troubleshooting

### Se Compaiono Spike di Val Loss
**Sintomo**: val_loss > 0.03 in alcune epoche

**Soluzione**:
1. Ridurre max_lr a 0.0006
2. Aumentare warmup_epochs a 60
3. Riduci gradient_clip a 0.3

### Se Bias Non Migliora
**Sintomo**: Buried bias ancora > +0.04, Exposed bias < -0.06

**Soluzione**:
1. Aumentare `exposed_weight` da 2.0 a 2.5
2. Aumentare `buried_weight` da 1.5 a 2.0
3. Aumentare `asymmetric_penalty` da 1.5 a 2.0

### Se Overfitting
**Sintomo**: train_loss << val_loss (gap > 30%)

**Soluzione**:
1. Aumentare dropout a 0.28-0.30
2. Aumentare weight_decay a 1.5e-4
3. Ridurre hidden_channels a 128

---

## Innovazioni Tecniche

### 1. Range-Specific Loss Implementation
File modificati:
- `src/training/train.py`: Classe `RangeSpecificWeightedMSELoss`
- `main.py`: Supporto per `loss_type='range_specific'`

### 2. Asymmetric Penalty
Penalty direzionale basato su:
```python
# Penalizza sovrastima di buried
if (target < 0.2) and (pred > target):
    weight *= asymmetric_penalty

# Penalizza sottostima di exposed
if (target >= 1.2) and (pred < target):
    weight *= asymmetric_penalty
```

---

## Confronto Fasi

| Aspetto | Phase 13a | Phase 14 | **Phase 15** |
|---------|-----------|----------|--------------|
| **Strategia** | Baseline stabile | Scaling up aggressivo | Ottimizzazione bilanciata |
| **Parametri** | 71K | 513K | 254K |
| **Layers** | 3 | 5 | 4 |
| **Hidden** | 128 | 176 | 136 |
| **Max LR** | 0.001 | 0.0009 | 0.0007 |
| **Gradient Clip** | 1.0 | 1.0 | 0.5 |
| **Warmup** | 35 | 40 | 50 |
| **Loss** | Exposure-weighted | Exposure-weighted | **Range-specific** |
| **Training Stability** | ✓ Stabile | ✗ Spike frequenti | **✓ Ottimizzato** |
| **Bias Handling** | ✗ Non affrontato | ✗ Non risolto | **✓ Targeted** |
| **ROI** | Baseline | Basso (+0.8%) | **Target: Alto (+1.5-2%)** |

---

## Files Generati dal Training

```
experiments/
├── checkpoints/
│   └── phase15_optimized/
│       ├── best_model.pt         # Miglior modello per R²
│       └── last_model.pt         # Ultimo checkpoint
└── logs/
    └── phase15_optimized/
        ├── training_history.csv  # Metriche per epoca
        ├── test_metrics.json     # Metriche finali test set
        └── [visualizations].png  # Grafici
```

---

## Conclusioni

Phase 15 rappresenta un **approccio bilanciato ed evidence-based**:

**✓ Vantaggi**:
1. Modello più compatto (254K vs 513K) → migliore fit al dataset
2. Training più stabile (LR ridotto, gradient clip tight, warmup lungo)
3. Loss innovativa che affronta il bias sistematico
4. Migliore ROI atteso (parametri/performance)

**⚠ Rischi**:
1. Loss range-specific potrebbe essere troppo aggressiva → monitorare overfitting
2. Modello più piccolo potrebbe avere capacità limitata → verificare R²
3. LR più basso potrebbe rallentare convergenza → verificare epoche necessarie

**Obiettivo**: R² > 0.895 con bias < ±0.03 e training stabile.

Buon training! 🚀
