# Phase 14: Large Model Configuration (~500K Parameters)

## Overview
Nuova configurazione del modello GNN con capacità drammaticamente aumentata rispetto a Phase 13a.

## Confronto Configurazioni

| Parametro | Phase 13a | Phase 14 | Variazione |
|-----------|-----------|----------|------------|
| **hidden_channels** | 128 | 175 | +36.7% |
| **num_layers** | 3 | 5 | +66.7% |
| **Parametri totali** | ~71,808 | **498,860** | **+594.8%** (7.0x) |
| **dropout** | 0.25 | 0.28 | +12% |
| **batch_size** | 32 | 28 | -12.5% |
| **max_lr** | 0.001 | 0.0009 | -10% |
| **weight_decay** | 1.0e-4 | 1.5e-4 | +50% |
| **warmup_epochs** | 35 | 40 | +14.3% |
| **early_stopping_patience** | 50 | 60 | +20% |

## Architettura del Modello

```
Input (93 features)
    ↓
Linear Projection (93 → 175)
    ↓
GATv2 Layer 1 (175 → 175, 4 heads) + BatchNorm + Dropout
    ↓
GATv2 Layer 2 (175 → 175, 4 heads) + BatchNorm + Dropout
    ↓
GATv2 Layer 3 (175 → 175, 4 heads) + BatchNorm + Dropout
    ↓
GATv2 Layer 4 (175 → 175, 4 heads) + BatchNorm + Dropout
    ↓
GATv2 Layer 5 (175 → 175, 4 heads) + BatchNorm + Dropout
    ↓
Output Projection (175 → 87 → 1)
```

**Totale: 498,860 parametri trainable**

### Breakdown Parametri per Layer (GATv2)

Ogni layer GATv2 contiene:
- `lin_l`: 175×175 = 30,625 params (left transformation)
- `lin_r`: 175×175 = 30,625 params (right transformation)
- `res`: 175×175 = 30,625 params (residual connection)
- `lin_edge`: 175×12 = 2,100 params (edge features)
- `att`, `bias`: ~350 params (attention mechanism)
- BatchNorm: 350 params (weight + bias)

**Total per layer: ~95,000 parametri**

## Modifiche per Stabilità

Dato l'aumento significativo di parametri, sono stati implementati diversi accorgimenti per la stabilità del training:

### 1. Regolarizzazione Aumentata
- **Dropout**: 0.28 (da 0.25) - previene overfitting con maggiore capacità del modello
- **Weight Decay**: 1.5e-4 (da 1.0e-4) - regolarizzazione L2 più forte

### 2. Learning Rate Più Conservativo
- **max_lr**: 0.0009 (da 0.001) - riduce oscillazioni durante training
- **warmup_epochs**: 40 (da 35) - warmup più graduale

### 3. Ottimizzazione Memoria
- **batch_size**: 28 (da 32) - riduce memory footprint su GPU

### 4. Maggiore Pazienza
- **early_stopping_patience**: 60 (da 50) - più tempo per convergenza

## Come Lanciare il Training

### 1. Preparazione
```bash
cd GNN_atom_exposure
```

### 2. Verifica Configurazione (opzionale)
```bash
.venv/Scripts/python.exe -c "from src.utils.config import Config; cfg = Config.from_yaml('configs/phase14_large_model.yaml'); print(f'Layers: {cfg[\"model\"][\"num_layers\"]}, Hidden: {cfg[\"model\"][\"hidden_channels\"]}')"
```

### 3. Verifica Parametri del Modello (opzionale)
```bash
.venv/Scripts/python.exe test_model_params.py
```
Output atteso: **498,860 parametri**

### 4. Lancio Training
```bash
.venv/Scripts/python.exe main.py --config configs/phase14_large_model.yaml
```

## Monitoraggio Durante il Training

### Metriche da Osservare

1. **Loss Convergence**
   - Train loss dovrebbe diminuire gradualmente
   - Val loss non dovrebbe oscillare troppo
   - Gap train/val loss indica overfitting (se troppo grande)

2. **R² Score**
   - Target: > 0.90 (miglioramento da 0.8817 di Phase 13a)
   - Monitorare sia train R² che validation R²

3. **Memory Usage**
   - GPU memory dovrebbe rimanere < 90% utilizzo
   - Se OOM error, ridurre batch_size a 24 o 20

4. **Training Speed**
   - Aspettarsi ~2-3x tempo per epoca rispetto a Phase 13a
   - Con 5 layer invece di 3, il forward pass è più lento

### Segnali di Problemi

| Problema | Sintomo | Soluzione |
|----------|---------|-----------|
| **Overfitting** | Val loss cresce mentre train loss scende | Aumentare dropout a 0.30-0.32 |
| **Oscillazioni** | Loss oscilla fortemente | Ridurre max_lr a 0.0007-0.0008 |
| **OOM Error** | Out of memory su GPU | Ridurre batch_size a 24/20 |
| **Underfit** | Entrambi loss alti | Aumentare num_epochs o max_lr |
| **Slow convergence** | Loss scende molto lentamente | Controllare learning rate schedule |

## Aspettative di Performance

### Con Dataset Attuale
- **R² atteso**: 0.90 - 0.92 (vs 0.8817 di Phase 13a)
- **MSE atteso**: Riduzione del 15-25% rispetto a Phase 13a
- **Tempo training**: ~2-3x rispetto a Phase 13a per epoca
- **Epoche necessarie**: 150-200 (simile a Phase 13a)

### Vantaggi della Configurazione
1. **5 layers**: Cattura relazioni gerarchiche più profonde nelle strutture proteiche
2. **175 channels**: Sufficiente capacità rappresentativa senza eccessiva complessità
3. **498K params**: Bilanciamento ottimale tra capacità espressiva e rischio overfitting
4. **GATv2**: Attention mechanism migliora aggregazione features da atomi vicini

### Svantaggi/Rischi
1. **Overfitting**: Con 7x parametri, rischio maggiore su dataset piccoli
2. **Training time**: Più lento rispetto a Phase 13a
3. **Memory**: Richiede più memoria GPU (batch_size ridotto)

## File Generati dal Training

Durante il training, i file saranno salvati in:

```
experiments/
├── checkpoints/
│   └── phase14_large_model/
│       ├── best_model.pt         # Miglior modello (per R²)
│       └── last_model.pt         # Ultimo checkpoint
└── logs/
    └── phase14_large_model/
        ├── train_log.csv         # Metriche per epoca
        └── tensorboard/          # TensorBoard logs (se abilitato)
```

## Prossimi Passi

Dopo il completamento del training:

1. **Valutazione**: Confrontare metriche con Phase 13a
2. **Analisi errori**: Identificare dove il modello fallisce
3. **Feature importance**: Analizzare quali features sono più importanti
4. **Hyperparameter tuning**: Se necessario, fine-tuning di dropout, lr, etc.

## Backup Configuration

In caso di problemi, è sempre possibile tornare a Phase 13a:
```bash
.venv/Scripts/python.exe main.py --config configs/phase13a_lower_lr.yaml
```
