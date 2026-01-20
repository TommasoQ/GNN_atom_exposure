# Edge Ablation Study for GATv2 Model

## Obiettivo

Questo studio di ablation verifica se il modello GATv2 (phase13a) utilizza effettivamente:
1. **La struttura del grafo** (connessioni tra atomi vicini)
2. **Le edge features** (12-dim: bond types, distanze, etc.)

O se invece si affida principalmente alle **node features** (come il sospetto `contact_count_10A`).

## Contesto

Analisi preliminari suggeriscono che il modello usa prevalentemente una singola feature che conta i vicini entro 10Å. Questo esperimento testa formalmente questa ipotesi.

## Approcci di Test

### 1. Normal (Baseline)
- Inferenza standard con grafo completo
- Tutte le edge features presenti
- Rappresenta le performance originali del modello

### 2. Self-Loops Only
- **Modifica**: `edge_index` contiene solo self-loops (ogni nodo collegato solo a se stesso)
- **Edge features**: Azzerate per i self-loops
- **Cosa testa**: Se il modello usa la struttura del grafo o solo le node features
- **Predizione**: Se il modello è "stupido", le performance rimarranno simili

### 3. Zero Edge Features
- **Modifica**: `edge_attr` impostato a zero per tutti gli archi
- **Struttura**: Grafo completo invariato
- **Cosa testa**: Se le edge features (bond types, distanze) contribuiscono
- **Predizione**: Se le edge features non contribuiscono, le performance rimarranno simili

## Struttura dei File

```
analysis/
├── evaluate_edge_ablation.py      # Script principale per inferenza
├── compare_ablation_results.py    # Script di confronto risultati
├── README.md                       # Questo file
└── results/                        # Directory risultati (creata automaticamente)
    ├── normal/
    │   ├── test_metrics.json
    │   ├── predictions_vs_actual.png
    │   └── error_distribution.png
    ├── self_loops/
    │   ├── test_metrics.json
    │   ├── predictions_vs_actual.png
    │   └── error_distribution.png
    ├── zero_features/
    │   ├── test_metrics.json
    │   ├── predictions_vs_actual.png
    │   └── error_distribution.png
    └── comparison/
        ├── metrics_comparison.json
        ├── metrics_table.txt
        └── metrics_comparison.png
```

## Come Eseguire

### Prerequisiti

1. Modello trainato disponibile (checkpoint)
2. Dataset test disponibile
3. Ambiente Python con dipendenze installate

### Step 1: Inferenza Baseline (Normal)

Dalla root del progetto (`D:\PythonProjects\Atom_exposure\GNN_atom_exposure\`):

```bash
python experiments/baselines/phase13a_best/analysis/evaluate_edge_ablation.py \
  --config configs/phase13a_lower_lr.yaml \
  --checkpoint experiments/baselines/phase13a_best/best_model.pt \
  --mode normal \
  --output-dir experiments/baselines/phase13a_best/analysis/results/normal
```

**Output atteso**:
- `test_metrics.json`: Metriche (MAE, RMSE, R², etc.)
- `predictions_vs_actual.png`: Scatter plot predizioni vs veri valori
- `error_distribution.png`: Distribuzione errori

### Step 2: Inferenza Solo Self-Loops

```bash
python experiments/baselines/phase13a_best/analysis/evaluate_edge_ablation.py \
  --config configs/phase13a_lower_lr.yaml \
  --checkpoint experiments/baselines/phase13a_best/best_model.pt \
  --mode self_loops \
  --output-dir experiments/baselines/phase13a_best/analysis/results/self_loops
```

### Step 3: Inferenza Edge Features Azzerate

```bash
python experiments/baselines/phase13a_best/analysis/evaluate_edge_ablation.py \
  --config configs/phase13a_lower_lr.yaml \
  --checkpoint experiments/baselines/phase13a_best/best_model.pt \
  --mode zero_features \
  --output-dir experiments/baselines/phase13a_best/analysis/results/zero_features
```

### Step 4: Confronto Risultati

```bash
python experiments/baselines/phase13a_best/analysis/compare_ablation_results.py \
  --results-dir experiments/baselines/phase13a_best/analysis/results \
  --output-dir experiments/baselines/phase13a_best/analysis/results/comparison
```

**Output**:
- Tabella comparativa stampata a console
- `metrics_comparison.json`: Confronto completo in formato JSON
- `metrics_table.txt`: Tabella salvata in formato testo
- `metrics_comparison.png`: Grafico a barre con MAE, RMSE, R²

## Interpretazione Risultati

### Scenario 1: Il Modello è "Stupido" (usa solo node features)

**Sintomi**:
- R² con `self_loops` ≈ R² con `normal` (differenza < 0.05)
- R² con `zero_features` ≈ R² con `normal` (differenza < 0.05)
- MAE e RMSE simili in tutte le modalità

**Conclusione**: Il modello ignora completamente:
- La struttura del grafo (connessioni tra atomi)
- Le edge features (bond types, distanze)

Usa solo le node features locali (probabilmente `contact_count_10A`).

### Scenario 2: Il Modello Usa la Struttura del Grafo

**Sintomi**:
- R² con `self_loops` << R² con `normal` (differenza significativa > 0.1)
- Performance degrada significativamente senza archi reali

**Conclusione**: Il modello sfrutta la connettività del grafo (chi è vicino a chi).

### Scenario 3: Il Modello Usa le Edge Features

**Sintomi**:
- R² con `zero_features` << R² con `normal` (differenza significativa > 0.1)
- Performance degrada quando le edge features sono azzerate

**Conclusione**: Il modello usa informazioni su bond types, distanze, etc.

### Scenario 4: Uso Misto

**Sintomi**:
- Leggera degradazione in entrambi i test (0.05 < differenza < 0.1)

**Conclusione**: Il modello usa sia node features che informazioni del grafo, ma non in modo critico.

## Esempio di Output

```
================================================================================
EDGE ABLATION STUDY - RESULTS COMPARISON
================================================================================

Metric               Normal          Self-Loops           Zero Features
----------------------------------------------------------------------------------------------------
MAE                  0.0543          0.0551 (+1.5%)       0.0548 (+0.9%)
RMSE                 0.0821          0.0834 (+1.6%)       0.0827 (+0.7%)
R²                   0.8817          0.8792 (-0.0025)     0.8805 (-0.0012)
Pearson Corr         0.9392          0.9378 (-0.0014)     0.9384 (-0.0008)
Median AE            0.0412          0.0419 (+1.7%)       0.0415 (+0.7%)
================================================================================

KEY FINDINGS:
----------------------------------------------------------------------------------------------------
1. Graph structure contributes MINIMALLY (R² change: -0.0025)
   → Model primarily uses node features, not graph connectivity

2. Edge features contribute MINIMALLY (R² change: -0.0012)
   → Model doesn't use edge features (bond types, distances, etc.)
================================================================================
```

## Opzioni Avanzate

### Script evaluate_edge_ablation.py

```bash
python evaluate_edge_ablation.py --help
```

Opzioni disponibili:
- `--config`: Path al file di configurazione YAML
- `--checkpoint`: Path al checkpoint del modello (required)
- `--mode`: Modalità ablation (normal/self_loops/zero_features, required)
- `--output-dir`: Directory output (default: auto)
- `--batch-size`: Override batch size
- `--num-workers`: Override numero workers per DataLoader
- `--device`: Device (cuda/cpu, default: auto)

### Script compare_ablation_results.py

```bash
python compare_ablation_results.py --help
```

Opzioni disponibili:
- `--results-dir`: Directory contenente i risultati (required)
- `--output-dir`: Directory output confronto (default: results-dir/comparison)

## Note Tecniche

### Implementazione Self-Loops

```python
def create_self_loops(num_nodes, device):
    edge_index = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([edge_index, edge_index], dim=0)
    return edge_index
```

Ogni nodo `i` ha un solo arco `(i, i)`.

### Implementazione Zero Features

```python
batch.edge_attr = torch.zeros_like(batch.edge_attr)
```

Tutte le 12 edge features vengono impostate a 0 mantenendo la forma.

### GATv2Conv e Self-Loops

Il modello GATv2 ha `add_self_loops=False` per default quando si usano edge features. Gli script gestiscono manualmente i self-loops per la modalità `self_loops`.

## Troubleshooting

### Errore: Checkpoint not found

Verifica che il path al checkpoint sia corretto:
```bash
ls experiments/baselines/phase13a_best/best_model.pt
```

### Errore: Config not found

Verifica che il config YAML esista:
```bash
ls configs/phase13a_lower_lr.yaml
```

### Memoria GPU insufficiente

Riduci il batch size:
```bash
python evaluate_edge_ablation.py ... --batch-size 16
```

Oppure usa CPU:
```bash
python evaluate_edge_ablation.py ... --device cpu
```

## Riferimenti

- **Config**: [configs/phase13a_lower_lr.yaml](../../../../configs/phase13a_lower_lr.yaml)
- **Modello**: [src/models/gnn.py](../../../../src/models/gnn.py)
- **Dataset**: [src/data/dataset_fixed.py](../../../../src/data/dataset_fixed.py)
- **Evaluation**: [src/training/evaluate.py](../../../../src/training/evaluate.py)

## Autore & Data

Esperimento creato per analizzare il contributo di edge features e struttura del grafo nel modello GATv2 phase13a.

Data creazione: 2026-01-20
