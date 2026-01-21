# Phase 17: Balanced Loss Fine-Tuning

## Overview
Phase 17 è una configurazione **balanced** basata sull'analisi del fallimento di Phase 16. L'obiettivo è trovare il giusto equilibrio tra riduzione del bias e mantenimento dell'R².

## Analisi del Problema (da Phase 16)

### Risultati Phase 16 - FALLIMENTO
- **R² Test**: 0.8828 (**-0.64%** vs Phase 15: 0.8892) ❌
- **Parametri**: 314K (+24% vs Phase 15: 254K)
- **Bias**: Mixed results - buried migliorato ma range intermedi peggiorati

### Cosa è Andato Storto in Phase 16

| Range | Phase 15 | Phase 16 | Δ | Problema |
|-------|----------|----------|---|----------|
| Buried (0-0.2) | +0.0394 | +0.0313 | -21% | ✓ Migliorato |
| **Semi-buried (0.2-0.5)** | +0.0120 | **-0.0220** | **Flipped** | ❌ PEGGIORATO |
| **Intermediate (0.5-0.8)** | -0.0166 | **-0.0385** | **2.3x** | ❌ PEGGIORATO |
| **Semi-exposed (0.8-1.2)** | -0.0396 | **-0.0592** | **1.5x** | ❌ PEGGIORATO |
| Exposed (1.2+) | -0.0603 | -0.0591 | -2% | ✓ Lieve migliore |

**Diagnosi:**
1. **Loss troppo aggressiva** (buried=2.0, exposed=2.5, asymmetric=2.0)
2. **Range intermedi non protetti** (semi_buried=1.0, intermediate=1.0)
3. **Trade-off pessimo**: -64 punti base di R² per -8 punti di buried bias
4. Il modello "sposta" l'errore invece di ridurlo globalmente

---

## Strategia Phase 17

### 1. Return to Phase 15 Model Size
**Motivazione**: Phase 16 ha usato 314K params, ma il problema era la loss, non la capacità

| Parametro | Phase 15 | Phase 16 | **Phase 17** |
|-----------|----------|----------|--------------|
| **hidden_channels** | 136 | 152 | **136** |
| **num_layers** | 4 | 4 | **4** |
| **dropout** | 0.26 | 0.27 | **0.26** |
| **Parametri totali** | 254K | 314K | **254K** |

**Perché**: Phase 15 aveva training stabile e buon R². Non serve più capacità, serve loss meglio calibrata.

### 2. Moderate Loss Weight Tuning

#### Strategia: Middle Ground tra Phase 15 e Phase 16

| Peso | Phase 15 | Phase 16 | **Phase 17** | Δ vs 15 | Logica |
|------|----------|----------|--------------|---------|--------|
| **buried** | 1.5 | 2.0 | **1.6** | +7% | Middle ground (1.5+2.0)/2 ≈ 1.75, ma più conservativo |
| **semi_buried** | 1.0 | 1.0 | **1.1** | +10% | **NEW protection** - previeni flipping |
| **intermediate** | 1.0 | 1.0 | **1.1** | +10% | **NEW protection** - previeni peggioramento |
| **semi_exposed** | 1.3 | 1.3 | **1.4** | +8% | Slight increase per balance |
| **exposed** | 2.0 | 2.5 | **2.2** | +10% | Middle ground (2.0+2.5)/2 = 2.25, arrotondato |
| **asymmetric_penalty** | 1.5 | 2.0 | **1.6** | +7% | Middle ground conservativo |

#### Effetto Atteso

**Esempio 1: Atomo Buried Sovrastimato**
- Target: 0.15, Pred: 0.20 (errore: +0.05)
- Phase 15: weight = 1.5 × 1.5 = 2.25x
- Phase 16: weight = 2.0 × 2.0 = 4.0x
- **Phase 17: weight = 1.6 × 1.6 = 2.56x** (+14% vs Phase 15, -36% vs Phase 16)

**Esempio 2: Atomo Exposed Sottostimato**
- Target: 1.40, Pred: 1.20 (errore: -0.20)
- Phase 15: weight = 2.0 × 1.5 = 3.0x
- Phase 16: weight = 2.5 × 2.0 = 5.0x
- **Phase 17: weight = 2.2 × 1.6 = 3.52x** (+17% vs Phase 15, -30% vs Phase 16)

**Esempio 3: Atomo Intermediate (NEW PROTECTION)**
- Target: 0.60, Pred: 0.65 (errore: +0.05)
- Phase 15: weight = 1.0 × 1.0 = 1.0x (nessuna protezione)
- Phase 16: weight = 1.0 × 1.0 = 1.0x (nessuna protezione, degradato)
- **Phase 17: weight = 1.1 × 1.0 = 1.1x** (+10% protezione base)

### 3. Training Hyperparameters
**Strategia**: Mantenere TUTTO identico a Phase 15 (provato stabile)

```yaml
# IDENTICO A PHASE 15
max_lr: 0.0007
gradient_clip: 0.5
warmup_epochs: 50
weight_decay: 1.2e-04
batch_size: 32
dropout: 0.26
```

---

## Configurazione Completa

### Model Architecture
```yaml
model:
  in_channels: 93
  hidden_channels: 136      # SAME AS PHASE 15
  num_layers: 4             # SAME AS PHASE 15
  dropout: 0.26             # SAME AS PHASE 15
  conv_type: gatv2
  edge_dim: 12

Total Parameters: 253,913 (~254K) - IDENTICAL TO PHASE 15
```

### Training Hyperparameters
```yaml
training:
  # Optimizer (IDENTICAL TO PHASE 15)
  learning_rate: 5.0e-04
  weight_decay: 1.2e-04
  gradient_clip: 0.5

  # Scheduler (IDENTICAL TO PHASE 15)
  scheduler: one_cycle
  max_lr: 0.0007
  warmup_epochs: 50
  div_factor: 25.0
  final_div_factor: 10000.0

  # Data (IDENTICAL TO PHASE 15)
  batch_size: 32
  num_epochs: 200

  # Early stopping (IDENTICAL TO PHASE 15)
  early_stopping_patience: 60
  early_stopping_metric: r2

  # BALANCED: Loss function (MODERATE tuning from Phase 15)
  weighted_loss: true
  loss_type: range_specific
  loss_range_weights:
    buried: 1.6              # +7% from 1.5 (vs +33% in Phase 16)
    semi_buried: 1.1         # NEW +10% protection
    intermediate: 1.1        # NEW +10% protection
    semi_exposed: 1.4        # +8% from 1.3
    exposed: 2.2             # +10% from 2.0 (vs +25% in Phase 16)
  loss_asymmetric_penalty: 1.6  # +7% from 1.5 (vs +33% in Phase 16)
```

---

## Come Lanciare il Training

### Lancia Training
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe main.py --config configs/phase17_balanced.yaml
```

### Output Atteso
```
Creating model...
  Architecture: GATV2
  Layers: 4, Hidden: 136
  Parameters: 253,913
  Loss: Range-Specific Weighted MSE
    - Buried (0-0.2): 1.6x, Semi-buried (0.2-0.5): 1.1x
    - Intermediate (0.5-0.8): 1.1x, Semi-exposed (0.8-1.2): 1.4x
    - Exposed (1.2+): 2.2x, Asymmetric penalty: 1.6x
  Scheduler: One Cycle (will initialize after data loading)
```

---

## Aspettative e Metriche di Successo

### Performance Target

**Best Case Scenario:**
| Metrica | Phase 15 | Phase 16 | **Phase 17 Best Case** |
|---------|----------|----------|------------------------|
| **R² Test** | 0.8892 | 0.8828 | **0.8895-0.8900** ✓ |
| Buried bias | +0.0394 | +0.0313 | **+0.025-0.030** ✓ |
| Semi-buried bias | +0.0120 | -0.0220 | **+0.005 to +0.015** ✓ |
| Intermediate bias | -0.0166 | -0.0385 | **-0.010 to -0.020** ✓ |
| Semi-exposed bias | -0.0396 | -0.0592 | **-0.030 to -0.040** ✓ |
| Exposed bias | -0.0603 | -0.0591 | **-0.045 to -0.055** ✓ |

**Acceptable Scenario:**
- R² ≥ 0.8890 (at least not worse than Phase 15)
- Buried bias: < +0.035 (some improvement)
- Exposed bias: < -0.055 (some improvement)
- **No degradation** in intermediate ranges

**Failure Scenario (need to revert to Phase 15):**
- R² < 0.8885 (worse than Phase 15)
- OR: Intermediate ranges degrade again

### Critical Success Metrics

**Primary (Must Have):**
1. **R² ≥ 0.8890** (at least match Phase 15)
2. **No degradation in intermediate ranges** (semi_buried, intermediate, semi_exposed should not get worse)

**Secondary (Nice to Have):**
3. Buried bias: -10% to -20% improvement (target: +0.030 to +0.035)
4. Exposed bias: -8% to -15% improvement (target: -0.050 to -0.055)

---

## Monitoraggio Durante il Training

### 1. Validation Loss Stability
```bash
# Check per spike
grep "val_loss" experiments/logs/phase17_balanced/training_history.csv | awk -F',' '$3 > 0.03 {print "Spike at epoch", $1, ":", $3}'
```
**Aspettativa**: Nessun spike (come Phase 15)

### 2. R² Tracking
```bash
# Check R² progression
grep "val_r2" experiments/logs/phase17_balanced/training_history.csv | tail -20
```
**Aspettativa**: Dovrebbe stabilizzarsi a ≥ 0.889

### 3. Confronto con Phase 15 alla Stessa Epoca
**Alla epoch 100-120** (tipicamente dove converge):
- Se val_r2 < 0.888: ⚠️ Warning - loss potrebbe essere ancora troppo aggressiva
- Se val_r2 ≥ 0.889: ✓ Good progress

---

## Decision Tree Post-Training

### Scenario 1: R² ≥ 0.8895 E Bias Migliorati E Range Intermedi Stabili
**Status**: ✓ **SUCCESS!**
**Action**:
- Accettare Phase 17 come best model
- Procedere con deployment e testing finale
- Documentare come optimal configuration

### Scenario 2: R² ≥ 0.8890 MA Bias Miglioramento Marginale (<5%)
**Status**: ⚠️ **Acceptable but not optimal**
**Action**:
- Confrontare con Phase 15 (R²=0.8892, bias medio)
- Se R² simile: preferire Phase 17 (bias leggermente migliore)
- Se R² peggiore: preferire Phase 15 (loss più semplice)
**Conclusion**: Probabilmente abbiamo raggiunto il limite architetturale

### Scenario 3: R² < 0.8890 O Range Intermedi Degradati
**Status**: ❌ **FAILURE**
**Action**:
- **Revert to Phase 15 as final model**
- Documentare che il bias trade-off non è favorevole
- Considerare approcci architetturali diversi se necessario migliorare bias

### Scenario 4: R² > 0.8900 MA Buried/Exposed Bias Ancora > Target
**Status**: ✓ **Good R² but bias improvable**
**Action**:
- Valutare se bias è "good enough" per deployment
- Se necessario: provare Phase 18 con weights leggermente più alti
  - buried: 1.6 → 1.7
  - exposed: 2.2 → 2.3
  - asymmetric: 1.6 → 1.7
- Ma solo se R² > 0.8900 (margine di sicurezza)

---

## Confronto Dettagliato Fasi

| Aspetto | Phase 15 | Phase 16 | **Phase 17** |
|---------|----------|----------|--------------|
| **Strategia** | First range-specific loss | Aggressive scaling | **Balanced tuning** |
| **Parametri** | 254K | 314K | **254K** |
| **Hidden** | 136 | 152 | **136** |
| **buried_weight** | 1.5 | 2.0 (+33%) | **1.6 (+7%)** |
| **semi_buried_weight** | 1.0 | 1.0 | **1.1 (NEW +10%)** |
| **intermediate_weight** | 1.0 | 1.0 | **1.1 (NEW +10%)** |
| **semi_exposed_weight** | 1.3 | 1.3 | **1.4 (+8%)** |
| **exposed_weight** | 2.0 | 2.5 (+25%) | **2.2 (+10%)** |
| **asymmetric_penalty** | 1.5 | 2.0 (+33%) | **1.6 (+7%)** |
| **R² Result** | 0.8892 | 0.8828 ❌ | **Target: ≥0.8890** |
| **Buried bias** | +0.0394 | +0.0313 | **Target: +0.025-0.035** |
| **Exposed bias** | -0.0603 | -0.0591 | **Target: -0.045-0.055** |
| **Intermediate ranges** | OK | Degraded ❌ | **Target: Stable** |

---

## Troubleshooting

### Se R² < 0.8890
**Diagnosi**: Loss ancora troppo aggressiva
**Fix possibili**:
1. Ridurre weights ulteriormente (buried=1.55, exposed=2.1, asymmetric=1.55)
2. Tornare a Phase 15 come modello finale
3. Accettare che il trade-off bias vs R² ha un limite

### Se Range Intermedi Degradano Di Nuovo
**Diagnosi**: Protezione 1.1x non sufficiente
**Fix possibili**:
1. Aumentare semi_buried/intermediate a 1.2
2. Ridurre buried/exposed per compensare
3. Considerare loss function diversa (es. Huber loss con range weights)

### Se Bias Non Migliora Affatto
**Diagnosi**: Incremento troppo piccolo, indistinguibile dal noise
**Fix possibili**:
1. Se R² > 0.8895: provare Phase 18 con weights +10% più alti
2. Se R² ≤ 0.8895: accettare Phase 15 come limite pratico

---

## Filosofia di Design

Phase 17 rappresenta un **approccio ingegneristico pragmatico**:

**Principi guida:**
1. **Non rompere ciò che funziona**: Torna al modello di Phase 15 (254K params, training stabile)
2. **Cambi incrementali**: Aumenti moderati (+7-10%) invece di aggressivi (+25-33%)
3. **Protezione proattiva**: Aggiungi pesi ai range intermedi PRIMA che degradino
4. **Bias vs R² balance**: L'obiettivo primario è R², bias è secondario

**Aspettative realistiche:**
- ✓ Miglioriamo leggermente bias senza degradare R²
- ✓ OR scopriamo che Phase 15 era già ottimale
- ✓ OR confermiamo che serve cambio architetturale

Tutti e tre gli outcome sono **utili per la comprensione del problema**.

---

## Files Generati dal Training

```
experiments/
├── checkpoints/
│   └── phase17_balanced/
│       ├── best_model.pt         # Miglior modello per R²
│       └── last_model.pt         # Ultimo checkpoint
└── logs/
    └── phase17_balanced/
        ├── training_history.csv  # Metriche per epoca
        ├── test_metrics.json     # Metriche finali test set
        └── [visualizations].png  # Grafici
```

---

## Conclusioni

Phase 17 è un **esperimento di calibrazione fine**:

**Se funziona:**
- Abbiamo trovato il sweet spot per loss weights
- Conferma che l'approccio range-specific è valido con tuning corretto

**Se non funziona:**
- Conferma che Phase 15 era già ottimale
- Dimostra che il trade-off bias vs R² ha limiti pratici
- Fornisce evidenza che serve cambio architetturale per ulteriori miglioramenti

**In entrambi i casi**, abbiamo imparato qualcosa di utile.

Target: R² ≥ 0.8890, bias leggermente migliorato, range intermedi protetti.

Buon training! 🎯
