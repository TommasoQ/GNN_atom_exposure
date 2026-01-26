# Phase 14 Analysis Scripts

Questi script di analisi sono stati aggiornati per utilizzare il modello **Phase 14** (513K parametri).

## Modifiche Applicate

### Parametri del Modello
Tutti gli script ora utilizzano la configurazione Phase 14:

```python
model = AtomExposureGNN(
    in_channels=92,
    hidden_channels=176,      # era 128 in Phase 13a
    num_layers=5,             # era 3 in Phase 13a
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.28              # era 0.25 in Phase 13a
)
```

### Checkpoint Path Predefinito
Il checkpoint predefinito ora punta a Phase 14:
```
experiments/checkpoints/phase14_large_model/best_model.pt
```

### Script Modificati

1. **[attention_analysis.py](attention_analysis.py)**
   - Parametri modello aggiornati
   - `all_attention_by_layer` ora supporta 5 layers (era 3)
   - Tutti i loop `range(3)` aggiornati a `range(5)`
   - Checkpoint path aggiornato

2. **[edge_importance.py](edge_importance.py)**
   - Parametri modello aggiornati
   - Checkpoint path aggiornato

3. **[feature_importance.py](feature_importance.py)**
   - Parametri modello aggiornati
   - Checkpoint path aggiornato

## Come Usare gli Script

### 1. Attention Analysis
Analizza i pesi di attenzione appresi da GATv2 nei 5 layer:

```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py

# Oppure specifica un checkpoint diverso
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py --checkpoint path/to/model.pt

# Analizza più proteine (default: 50)
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py --num-proteins 100
```

**Output generati:**
- `experiments/analysis/attention_analysis/attention_statistics.csv` - Statistiche per layer/head
- `experiments/analysis/attention_analysis/attention_distribution.png` - Distribuzione pesi
- `experiments/analysis/attention_analysis/attention_vs_distance.png` - Correlazione distanza
- `experiments/analysis/attention_analysis/attention_by_edge_type.png` - Per tipo di edge
- `experiments/analysis/attention_analysis/attention_by_exposure.png` - Per livello esposizione
- `experiments/analysis/attention_analysis/attention_heatmap.png` - Heatmap (se trova proteine piccole)

### 2. Edge Feature Importance
Calcola l'importanza delle 12 edge features tramite zero-out method:

```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/edge_importance.py

# Specifica output directory
.venv/Scripts/python.exe experiments/analysis/edge_importance.py --output-dir results/edge_analysis
```

**Output generati:**
- `experiments/analysis/edge_importance.csv` - Importanza per feature
- `experiments/analysis/edge_importance.png` - Grafico a barre

**Edge features analizzate:**
1. bond_covalent, bond_peptide, bond_hydrophobic, bond_aromatic, bond_hbond, bond_ionic, bond_ring
2. distance, bond_length
3. normalized_dist, relative_dist
4. in_radius (radius graph indicator)

### 3. Feature Importance
Calcola l'importanza delle 93 node features tramite permutation importance:

```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/feature_importance.py

# Con più ripetizioni per stabilità (default: 3)
.venv/Scripts/python.exe experiments/analysis/feature_importance.py --n-repeats 5
```

**Output generati:**
- `experiments/analysis/feature_importance.csv` - Importanza per feature (tutte le 93)
- `experiments/analysis/feature_importance.png` - Top 30 features

**Feature groups analizzate:**
1. Numerical (24): polarità, carica, volume, accessibilità, etc.
2. Atom types (31): C.ar, N.am, O.2, etc.
3. Elements (5): C, N, O, S, Other
4. Residues (21): ALA, GLY, VAL, etc.
5. Geometric (8): mean_dist, contact_count, radial_position, etc.
6. Backbone angles (4): sin_phi, cos_phi, sin_psi, cos_psi

## Note sulla Compatibilità

### Differenze tra Phase 13a e Phase 14

| Parametro | Phase 13a | Phase 14 | Impatto |
|-----------|-----------|----------|---------|
| hidden_channels | 128 | 176 | Più capacità rappresentativa |
| num_layers | 3 | 5 | Più profondità gerarchica |
| dropout | 0.25 | 0.28 | Più regolarizzazione |
| **Parametri totali** | ~71K | ~513K | **7.2x più grande** |

### Attenzione

⚠️ **Questi script non sono retrocompatibili con checkpoint di Phase 13a!**

Se vuoi analizzare un modello Phase 13a, modifica i parametri del modello negli script o usa la versione originale degli script.

### Checkpoint Mancante

Se il checkpoint Phase 14 non esiste ancora:
```bash
# Specifica manualmente il checkpoint da analizzare
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py \
    --checkpoint experiments/checkpoints/phase13a_lower_lr/best_model.pt
```

Ma assicurati di modificare i parametri del modello nello script per corrispondere al checkpoint che stai usando!

## Tempi di Esecuzione Stimati

Con Phase 14 (513K params) i tempi sono ~2x più lunghi rispetto a Phase 13a:

- **attention_analysis.py** (50 proteine): ~10-15 minuti su CPU, ~3-5 minuti su GPU
- **edge_importance.py**: ~5-8 minuti su CPU, ~2-3 minuti su GPU
- **feature_importance.py** (3 repeats): ~30-45 minuti su CPU, ~10-15 minuti su GPU

## Troubleshooting

### "RuntimeError: Error(s) in loading state_dict"
Il checkpoint non corrisponde ai parametri del modello. Verifica:
1. Il checkpoint è davvero Phase 14? (513K params)
2. I parametri nello script corrispondono alla configurazione del checkpoint

### "FileNotFoundError: checkpoint not found"
Il checkpoint Phase 14 non esiste ancora. Opzioni:
1. Completa prima il training di Phase 14
2. Usa un checkpoint diverso con `--checkpoint path/to/model.pt`
3. Modifica il default negli script

### "CUDA out of memory"
Il modello Phase 14 usa più memoria. Soluzioni:
1. Riduci `--num-proteins` (per attention_analysis)
2. Usa batch size più piccolo (modificare gli script)
3. Esegui su CPU (più lento ma funziona sempre)

## Prossimi Passi

Dopo aver eseguito gli script, confronta i risultati con Phase 13a per capire:
1. **Attention patterns**: I 5 layer hanno pattern diversi rispetto ai 3 di Phase 13a?
2. **Feature importance**: Le feature più importanti sono cambiate?
3. **Edge importance**: Le edge features hanno impatti diversi?

Questi insight possono guidare:
- Ulteriori ottimizzazioni architetturali
- Feature engineering mirato
- Comprensione del perché Phase 14 ha performance migliori (o peggiori)
