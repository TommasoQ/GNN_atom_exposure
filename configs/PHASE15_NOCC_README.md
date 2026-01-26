# Phase 15 noCC: Re-training Best Model WITHOUT contact_count

## Overview

**Phase 15 noCC** è una ri-esecuzione esatta di Phase 15 (il miglior modello finora) ma **senza la feature `contact_count_10A`**.

### Motivazione
- **Analisi feature importance**: `contact_count_10A` risultava troppo dominante/pesante rispetto alle altre feature geometriche
- **Obiettivo**: Verificare se la rimozione migliora, peggiora o lascia invariate le performance

---

## Differenze tra Phase 15 e Phase 15 noCC

| Aspetto | Phase 15 (original) | **Phase 15 noCC** |
|---------|---------------------|-------------------|
| **Features totali** | 93 | **92** |
| **Geometric features** | 8 (con contact_count) | **7 (senza contact_count)** |
| **Parametri** | 253,913 | **253,777** (-136) |
| **Architettura** | hidden=136, layers=4 | **IDENTICA** |
| **Hyperparameters** | All settings | **IDENTICI** |
| **Loss** | Range-specific | **IDENTICA** |
| **Seed** | 42 | **IDENTICO** |

**UNICA DIFFERENZA**: Feature `geom_contact_count_10A` rimossa.

---

## Phase 15 Original Results (Baseline per Confronto)

```
R² Test:            0.8892
MAE:                ~0.087
Parametri:          253,913
Training:           Stabile, no spikes
Early stopping:     ~150-180 epochs

Bias per Range:
  Buried (0-0.2):      +0.0394
  Semi-buried (0.2-0.5): +0.0120
  Intermediate (0.5-0.8): -0.0166
  Semi-exposed (0.8-1.2): -0.0396
  Exposed (1.2+):        -0.0603
```

**Status**: Miglior modello finora (R² più alto, training stabile, bias ridotto vs Phase 14)

---

## Aspettative per Phase 15 noCC

### Scenario 1: Performance Equivalente (Most Likely)
**R² ≈ 0.8890-0.8895** (differenza < 0.0003)

**Interpretazione**: `contact_count` era **ridondante**
- Altre feature geometriche (es. `radial_position`) già catturano informazione di burial
- Buona notizia: semplificazione senza perdita di performance

### Scenario 2: Performance Migliorata
**R² > 0.8895** (differenza > 0.0003)

**Interpretazione**: `contact_count` era **dannoso**
- Possibile overfitting o correlazione eccessiva con altre features
- Eccellente notizia: rimozione migliora il modello!

### Scenario 3: Performance Peggiorata (Less Likely)
**R² < 0.8885** (differenza > -0.0007)

**Interpretazione**: `contact_count` era **importante**
- La feature contribuiva in modo unico alla predizione
- Considerare di ripristinarla o investigare perché era importante

---

## Come Lanciare

### Prerequisiti
1. **Cancellare cache processed** (feature cambiate da 93 a 92):
   ```bash
   cd GNN_atom_exposure
   rm -rf dataset/processed/
   # Oppure backup:
   mv dataset/processed/ dataset/processed_93_backup/
   ```

2. **Verificare setup**:
   ```bash
   .venv/Scripts/python.exe verify_92_features.py
   ```

### Lanciare Training
```bash
cd GNN_atom_exposure
.venv\Scripts\python.exe main.py --config configs/phase15_noCC.yaml
```

### Output Atteso
```
Creating model...
  Architecture: GATV2
  Layers: 4, Hidden: 136
  Parameters: 253,777
  Loss: Range-Specific Weighted MSE
    - Buried (0-0.2): 1.5x, Semi-buried (0.2-0.5): 1.0x
    - Intermediate (0.5-0.8): 1.0x, Semi-exposed (0.8-1.2): 1.3x
    - Exposed (1.2+): 2.0x, Asymmetric penalty: 1.5x
  Scheduler: One Cycle (will initialize after data loading)

Loading dataset...
Dataset: 4556 proteins, 92 features (reduced=False, atom_type=True, geometric=True, backbone=True)
```

---

## Confronto Post-Training

### Metriche da Confrontare

1. **R² Test** (metrica primaria)
   ```bash
   # Phase 15 original
   grep "R² Score" experiments/logs/phase15_optimized/test_metrics.json
   # Phase 15 noCC
   grep "R² Score" experiments/logs/phase15_noCC/test_metrics.json
   ```

2. **Bias per Range**
   ```bash
   # Controllare nell'output finale:
   # Error by Exposure Range
   ```

3. **Training Stability**
   ```bash
   # Verificare che non ci siano spike:
   grep "val_loss" experiments/logs/phase15_noCC/training_history.csv | awk -F',' '$3 > 0.03'
   ```

4. **Convergence Speed**
   ```bash
   # A quale epoca ha fatto early stopping?
   tail experiments/logs/phase15_noCC/training_history.csv
   ```

### Tabella di Confronto

| Metrica | Phase 15 (93 feat) | Phase 15 noCC (92 feat) | Δ | Status |
|---------|-------------------|-------------------------|---|--------|
| **R² Test** | 0.8892 | ___ | ___ | ___ |
| **MAE** | ~0.087 | ___ | ___ | ___ |
| **Bias Buried** | +0.0394 | ___ | ___ | ___ |
| **Bias Exposed** | -0.0603 | ___ | ___ | ___ |
| **Early Stop Epoch** | ~150-180 | ___ | ___ | ___ |
| **Training Spikes** | None | ___ | ___ | ___ |

---

## Decision Tree Post-Training

### Caso 1: R² noCC ≈ R² original (±0.0003)
**Conclusione**: contact_count era **ridondante**

**Azioni**:
- ✅ **Usa Phase 15 noCC come best model** (più semplice)
- ✅ Continua futuri training con 92 features
- ✅ Aggiorna documentazione che 92 è lo standard

### Caso 2: R² noCC > R² original (+0.0003 o più)
**Conclusione**: contact_count era **dannoso**

**Azioni**:
- ✅ **Usa Phase 15 noCC come best model**
- ✅ Tutti i futuri training con 92 features
- ✅ Documenta che la rimozione ha migliorato il modello
- 🔍 Analizza perché contact_count causava problemi (correlazione con altre features?)

### Caso 3: R² noCC < R² original (-0.0007 o peggio)
**Conclusione**: contact_count era **importante**

**Azioni**:
- ⚠️ **Considera di ripristinare contact_count** (revert to 93 features)
- 🔍 Analizza feature importance dettagliata per capire perché era cruciale
- 🔍 Verifica se altre feature geometriche possono compensare
- Opzioni:
  - A) Ripristina 93 features
  - B) Continua con 92 ma accetta il trade-off
  - C) Aggiungi feature geometrica diversa ma meno dominante

---

## Files Generati

```
experiments/
├── checkpoints/
│   └── phase15_noCC/
│       ├── best_model.pt         # Miglior modello per R²
│       └── last_model.pt         # Ultimo checkpoint
└── logs/
    └── phase15_noCC/
        ├── training_history.csv  # Metriche per epoca
        ├── test_metrics.json     # Metriche finali test set
        └── [visualizations].png  # Grafici predizioni, errori, etc.
```

---

## Compatibilità con Checkpoints

### Loading Phase 15 noCC
```python
model = AtomExposureGNN(
    in_channels=92,  # IMPORTANTE: 92, non 93!
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26
)
checkpoint = torch.load('experiments/checkpoints/phase15_noCC/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### ⚠️ Incompatibilità
- **Phase 15 noCC checkpoint** (92 features) **NON può** essere usato con dataset 93 features
- **Phase 15 original checkpoint** (93 features) **NON può** essere usato con dataset 92 features
- Se serve confronto diretto: mantenere entrambe le versioni del dataset processed

---

## Troubleshooting

### Error: "RuntimeError: size mismatch for convs.0.lin_l.weight"
**Causa**: Checkpoint 93 features caricato con modello 92 features (o viceversa)

**Soluzione**: Verifica `in_channels` nel modello corrisponda al checkpoint

### Error: "RuntimeError: Expected input with 93 features, got 92"
**Causa**: Dataset processato con 93 features, modello configurato per 92

**Soluzione**: Cancella `dataset/processed/` e riprocessa

### Training più lento del previsto
**Causa**: Dataset si sta riprocessando (prima volta con 92 features)

**Soluzione**: Normale, solo la prima volta. Successive run useranno la cache.

---

## Conclusioni

Phase 15 noCC è un **esperimento di ablation** per validare l'importanza di `contact_count_10A`.

**Obiettivo**: Capire se la rimozione della feature dominante:
- 🟢 Migliora il modello (feature era dannosa)
- 🟡 Non cambia nulla (feature era ridondante)
- 🔴 Peggiora il modello (feature era importante)

In tutti i casi, otteniamo **informazione preziosa** sull'architettura del feature set.

**Next Steps**:
1. Lanciare training Phase 15 noCC
2. Confrontare metriche con Phase 15 original
3. Decidere quale versione (92 o 93 features) usare come standard
4. Documentare il risultato per futuri lavori

Buon training! 🚀
