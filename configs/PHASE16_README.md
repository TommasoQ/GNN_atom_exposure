# Phase 16: Enhanced Loss + Moderate Model Scaling

## Overview
Phase 16 è una configurazione **enhanced** basata sull'analisi dei risultati di Phase 15. L'obiettivo è continuare a ridurre il bias sistematico con pesi di loss più forti e un modello leggermente più grande per maggiore capacità.

## Analisi del Problema (da Phase 15)

### Risultati Phase 15
- **R² Test**: 0.8892 (+0.4% vs Phase 14: 0.8888, +0.9% vs Phase 13a: 0.8817)
- **Parametri**: 254K (50% vs Phase 14: 513K)
- **ROI**: Migliorato rispetto a Phase 14
- **Training Stability**: ✓ Molto migliorato (pochi spike isolati)

### Progressi Ottenuti
Il **range-specific loss** di Phase 15 ha funzionato parzialmente:

| Range | Phase 14 Bias | Phase 15 Bias | **Miglioramento** |
|-------|---------------|---------------|-------------------|
| Buried (0-0.2) | +0.0565 | +0.0394 | **-30%** ✓ |
| Semi-buried (0.2-0.5) | +0.0121 | +0.0120 | Stabile |
| Intermediate (0.5-0.8) | -0.0240 | -0.0166 | +31% ✓ |
| Semi-exposed (0.8-1.2) | -0.0478 | -0.0396 | +17% ✓ |
| Exposed (1.2+) | -0.0803 | -0.0603 | **-25%** ✓ |

**Validazione dell'approccio**: ✓ Il range-specific loss funziona!

### Problemi Rimanenti

#### 1. Bias Ancora Sopra il Target
Nonostante i miglioramenti, il bias è ancora significativo:
- **Buried**: +0.0394 (target: < +0.02)
- **Exposed**: -0.0603 (target: < -0.03)

**Gap rimanente**:
- Buried: +0.0194 sopra il target (~2x)
- Exposed: -0.0303 sopra il target (~2x)

#### 2. Capacità del Modello Potenzialmente Limitata
Phase 15 ha ridotto drasticamente i parametri (254K vs 513K):
- **Pro**: Training più stabile, migliore ROI
- **Contro**: Potrebbe essere troppo piccolo per catturare pattern complessi
- **Osservazione**: R² migliorato solo marginalmente (+0.0004 vs Phase 14)

---

## Strategia Phase 16

### 1. Moderate Model Scaling (+18%)
**Obiettivo**: Aumentare moderatamente la capacità del modello mantenendo stabilità

| Parametro | Phase 15 | Phase 16 | Δ |
|-----------|----------|----------|---|
| **num_layers** | 4 | 4 | - (stabile) |
| **hidden_channels** | 136 | 152 | +12% |
| **dropout** | 0.26 | 0.27 | +3.8% |
| **Parametri totali** | 254K | **~300K** | **+18%** |

**Motivazione**:
- 254K params → ~4 atomi/parametro
- 300K params → ~3.3 atomi/parametro (più capacità rappresentativa)
- Scaling moderato (18%) evita ritorno ai problemi di Phase 14
- 4 layers mantenuti (depth provata stabile)

### 2. Enhanced Range-Specific Loss Weights

#### Current Weights (Phase 15) → New Weights (Phase 16)

| Range | Phase 15 | Phase 16 | Δ | Motivazione |
|-------|----------|----------|---|-------------|
| **buried** | 1.5 | **2.0** | +33% | Bias +0.0394 ancora troppo alto |
| **semi_buried** | 1.0 | 1.0 | - | Bias +0.0120 accettabile |
| **intermediate** | 1.0 | 1.0 | - | Bias -0.0166 accettabile |
| **semi_exposed** | 1.3 | 1.3 | - | Bias -0.0396 migliorato ma monitorare |
| **exposed** | 2.0 | **2.5** | +25% | Bias -0.0603 ancora troppo alto |
| **asymmetric_penalty** | 1.5 | **2.0** | +33% | Rafforza penalty direzionale |

#### Effetto Atteso

**Esempio 1: Atomo Buried Sovrastimato**
- Target: 0.15, Pred: 0.20 (errore: +0.05)
- Phase 15: weight = 1.5 × 1.5 = 2.25x
- **Phase 16: weight = 2.0 × 2.0 = 4.0x** (+78% penalty)

**Esempio 2: Atomo Exposed Sottostimato**
- Target: 1.40, Pred: 1.20 (errore: -0.20)
- Phase 15: weight = 2.0 × 1.5 = 3.0x
- **Phase 16: weight = 2.5 × 2.0 = 5.0x** (+67% penalty)

**Aspettativa**: Riduzione più aggressiva del bias direzionale

### 3. Training Hyperparameters
**Strategia**: Mantenere l'approccio conservativo di Phase 15 (provato stabile)

```yaml
# KEPT from Phase 15 (tutti stabili e validati)
max_lr: 0.0007
gradient_clip: 0.5
warmup_epochs: 50
weight_decay: 1.2e-04
batch_size: 32

# SLIGHT adjustment for larger model
dropout: 0.27  # da 0.26 (+3.8%)
```

---

## Configurazione Completa

### Model Architecture
```yaml
model:
  in_channels: 93
  hidden_channels: 152      # +12% vs Phase 15 (136)
  num_layers: 4             # KEPT (proven stable)
  dropout: 0.27             # +3.8% vs Phase 15 (0.26)
  conv_type: gatv2
  edge_dim: 12

Total Parameters: ~300,000 (+18% vs Phase 15: 254K)
```

### Training Hyperparameters
```yaml
training:
  # Optimizer (KEPT from Phase 15)
  learning_rate: 5.0e-04
  weight_decay: 1.2e-04
  gradient_clip: 0.5

  # Scheduler (KEPT from Phase 15)
  scheduler: one_cycle
  max_lr: 0.0007
  warmup_epochs: 50
  div_factor: 25.0
  final_div_factor: 10000.0

  # Data (KEPT from Phase 15)
  batch_size: 32
  num_epochs: 200

  # Early stopping (KEPT from Phase 15)
  early_stopping_patience: 60
  early_stopping_metric: r2

  # ENHANCED: Loss function
  weighted_loss: true
  loss_type: range_specific
  loss_range_weights:
    buried: 2.0              # +33% from 1.5
    semi_buried: 1.0         # KEPT
    intermediate: 1.0        # KEPT
    semi_exposed: 1.3        # KEPT
    exposed: 2.5             # +25% from 2.0
  loss_asymmetric_penalty: 2.0  # +33% from 1.5
```

---

## Come Lanciare il Training

### 1. Verifica Configurazione (Opzionale)
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe -c "from src.utils.config import Config; cfg = Config.from_yaml('configs/phase16_enhanced.yaml'); print('Config OK')"
```

### 2. Lancia Training
```bash
.venv/Scripts/python.exe main.py --config configs/phase16_enhanced.yaml
```

### 3. Output Atteso
```
Creating model...
  Architecture: GATV2
  Layers: 4, Hidden: 152
  Parameters: ~300,000
  Loss: Range-Specific Weighted MSE
    - Buried (0-0.2): 2.0x, Semi-buried (0.2-0.5): 1.0x
    - Intermediate (0.5-0.8): 1.0x, Semi-exposed (0.8-1.2): 1.3x
    - Exposed (1.2+): 2.5x, Asymmetric penalty: 2.0x
  Scheduler: One Cycle (will initialize after data loading)
```

---

## Aspettative e Metriche di Successo

### Performance Target
| Metrica | Phase 13a | Phase 14 | Phase 15 | **Phase 16 Target** |
|---------|-----------|----------|----------|---------------------|
| **R² Test** | 0.8817 | 0.8888 | 0.8892 | **0.895-0.900** |
| **Parametri** | 71K | 513K | 254K | 300K |
| **ROI (vs 13a)** | Baseline | +0.8% | +0.9% | **+1.2-1.5%** |

### Bias Reduction Target (Primary Goal)
| Range | Phase 15 Bias | **Phase 16 Target** | **Improvement Needed** |
|-------|---------------|---------------------|------------------------|
| Buried (0-0.2) | +0.0394 | **< +0.02** | -50% |
| Exposed (1.2+) | -0.0603 | **< -0.03** | -50% |

### Training Stability
- **NO spike di validation loss** (>0.03)
- Convergenza smooth senza oscillazioni
- Early stopping tra epoch 130-180

---

## Monitoraggio Durante il Training

### Metriche Chiave da Osservare

#### 1. Validation Loss Stability
```bash
# Check per spike
grep "val_loss" experiments/logs/phase16_enhanced/training_history.csv | awk -F',' '$3 > 0.03 {print "Spike at epoch", $1, ":", $3}'
```
**Successo**: Nessun spike > 0.03 (come Phase 15)

#### 2. Bias per Range (End of Training)
Nell'output finale, controllare:
```
Error by Exposure Range:
Buried (0-0.2):      Bias: +0.0XXX  ← Target: < +0.02 (50% reduction from +0.0394)
Exposed (1.2+):      Bias: -0.0XXX  ← Target: < -0.03 (50% reduction from -0.0603)
```

#### 3. Training vs Validation Gap
```
Final epoch:
train_loss: ~0.014-0.015
val_loss:   ~0.016-0.017
Gap: < 20%
```
**Successo**: Gap moderato (no overfitting nonostante modello più grande)

#### 4. R² Improvement
```
Test R²: > 0.895 (almeno +0.0003 vs Phase 15: 0.8892)
```

---

## Troubleshooting

### Se Bias Non Migliora Abbastanza
**Sintomo**: Buried bias ancora > +0.025, Exposed bias < -0.04

**Diagnosi possibili**:
1. **Loss weights ancora troppo bassi** → Aumentare ulteriormente
2. **Modello potrebbe aver bisogno di più capacità** → Aumentare hidden_channels
3. **Dataset potrebbe avere bias intrinseco** → Analizzare distribuzione target

**Soluzioni**:
1. Aumentare `exposed_weight` da 2.5 a 3.0
2. Aumentare `buried_weight` da 2.0 a 2.5
3. Aumentare `asymmetric_penalty` da 2.0 a 2.5

### Se Compaiono Spike di Val Loss
**Sintomo**: val_loss > 0.03 in alcune epoche (improbabile con max_lr=0.0007)

**Soluzione**:
1. Ridurre max_lr a 0.0006
2. Aumentare warmup_epochs a 60
3. Ridurre gradient_clip a 0.4

### Se Overfitting
**Sintomo**: train_loss << val_loss (gap > 30%)

**Soluzione**:
1. Aumentare dropout a 0.29-0.30
2. Aumentare weight_decay a 1.5e-4
3. Ridurre hidden_channels a 144

### Se R² Non Migliora
**Sintomo**: R² < 0.893 (stesso o peggio di Phase 15)

**Diagnosi possibili**:
1. **Loss troppo aggressiva** → Modello sacrifica R² per ridurre bias
2. **Modello ancora troppo piccolo** → Serve più capacità
3. **Training non convergente** → Serve più epoche

**Soluzioni**:
1. Ridurre leggermente i pesi (`exposed: 2.3`, `buried: 1.8`, `asymmetric: 1.8`)
2. Aumentare hidden_channels a 160
3. Aumentare patience a 80 e verificare se converge più tardi

---

## Innovazioni Tecniche

### 1. Balanced Scaling Strategy
**Filosofia**: Scaling moderato e mirato
- Non ripetere l'errore di Phase 14 (7x scaling, +0.8% ROI)
- Non rimanere troppo conservativi come Phase 15 (bias ancora alto)
- **Sweet spot**: +18% params, loss weights +25-33%

### 2. Enhanced Asymmetric Penalty
**Meccanismo**: Penalty direzionale più forte
```python
# Phase 15
if (target < 0.2) and (pred > target):
    weight *= 1.5  # Total: 1.5 × 1.5 = 2.25x

# Phase 16
if (target < 0.2) and (pred > target):
    weight *= 2.0  # Total: 2.0 × 2.0 = 4.0x (+78%)
```

### 3. Selective Weight Increase
**Strategia**: Aumentare solo i pesi critici
- `buried` e `exposed`: +25-33% (bias ancora alto)
- `semi_buried`, `intermediate`, `semi_exposed`: KEPT (bias già OK)
- Evita di destabilizzare range già ben calibrati

---

## Confronto Fasi

| Aspetto | Phase 14 | Phase 15 | **Phase 16** |
|---------|----------|----------|--------------|
| **Strategia** | Scaling up aggressivo | Ottimizzazione bilanciata | Enhanced loss + moderate scaling |
| **Parametri** | 513K | 254K | **300K** |
| **Layers** | 5 | 4 | 4 |
| **Hidden** | 176 | 136 | **152** |
| **Max LR** | 0.0009 | 0.0007 | 0.0007 |
| **Gradient Clip** | 1.0 | 0.5 | 0.5 |
| **Warmup** | 40 | 50 | 50 |
| **Loss buried** | 1.5 | 1.5 | **2.0** |
| **Loss exposed** | 2.0 | 2.0 | **2.5** |
| **Asymmetric penalty** | 1.5 | 1.5 | **2.0** |
| **Training Stability** | ✗ Spike frequenti | ✓ Molto stabile | **✓ Atteso stabile** |
| **Bias Buried** | +0.0565 | +0.0394 | **Target: < +0.02** |
| **Bias Exposed** | -0.0803 | -0.0603 | **Target: < -0.03** |
| **R²** | 0.8888 | 0.8892 | **Target: 0.895-0.900** |

---

## Files Generati dal Training

```
experiments/
├── checkpoints/
│   └── phase16_enhanced/
│       ├── best_model.pt         # Miglior modello per R²
│       └── last_model.pt         # Ultimo checkpoint
└── logs/
    └── phase16_enhanced/
        ├── training_history.csv  # Metriche per epoca
        ├── test_metrics.json     # Metriche finali test set
        └── [visualizations].png  # Grafici
```

---

## Analisi dei Risultati (Post-Training)

### Script di Analisi Disponibili

**Nota**: Gli script di analisi sono configurati per Phase 14. Prima di usarli per Phase 16, modifica i parametri del modello negli script:

```python
# In attention_analysis.py, edge_importance.py, feature_importance.py
model = AtomExposureGNN(
    in_channels=93,
    hidden_channels=152,      # CAMBIA da 176 (Phase 14)
    num_layers=4,             # CAMBIA da 5 (Phase 14)
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.27              # CAMBIA da 0.28 (Phase 14)
)
```

Poi esegui:
```bash
# Attention analysis
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py \
    --checkpoint experiments/checkpoints/phase16_enhanced/best_model.pt

# Edge feature importance
.venv/Scripts/python.exe experiments/analysis/edge_importance.py \
    --checkpoint experiments/checkpoints/phase16_enhanced/best_model.pt

# Node feature importance
.venv/Scripts/python.exe experiments/analysis/feature_importance.py \
    --checkpoint experiments/checkpoints/phase16_enhanced/best_model.pt
```

---

## Conclusioni

Phase 16 rappresenta un **approccio incrementale ed evidence-based**:

**✓ Vantaggi**:
1. **Loss più aggressiva** testata e validata da Phase 15
2. **Modello leggermente più grande** (300K) per maggiore capacità senza rischio overfitting
3. **Training stabile** mantenuto da Phase 15
4. **Target chiari e misurabili** per bias reduction

**⚠ Rischi**:
1. Loss troppo aggressiva potrebbe sacrificare R² globale per ridurre bias
2. Modello più grande potrebbe non migliorare abbastanza (ROI decrescente)
3. Potremmo aver raggiunto il limite di questa architettura

**Decision Tree Post-Training**:

```
Se Bias < ±0.02 e R² > 0.895:
    ✓ SUCCESS! Architettura ottimale trovata
    → Preparare deployment e testing finale

Se Bias < ±0.025 ma R² < 0.893:
    ⚠ Loss troppo aggressiva
    → Phase 17: Ridurre weights leggermente, focus su R²

Se Bias > ±0.03:
    ⚠ Loss non abbastanza forte
    → Phase 17: Aumentare weights ulteriormente o provare architecture change

Se Training instabile:
    ⚠ Modello troppo grande o LR troppo alto
    → Ridurre hidden_channels o max_lr
```

**Obiettivo**: Bias < ±0.02, R² > 0.895, training stabile.

Buon training! 🚀
