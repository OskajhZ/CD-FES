# Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition

Code & logs for the article:

> Lixian Zhu, Xiangnan Zhang, Ranqi Lu, Jingyu Liu, Fuze Tian, Juan Wang, Jingxin Liu, and Bin Hu,  
> *"Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition"* .

## Structure

```
CD-FES/
├── SourceFiles/       # All executable code, organised by dataset
└── ExperimentLogs/    # Raw logs corresponding to the results in the paper
```

### SourceFiles

| Dataset | Path | Content |
|---------|------|---------|
| **MODMA** | `SourceFiles/MODMA/` | Preprocessing (128ch → FP1/FPz/FP2), dynamic modeling (LSTM/GRU/MiniRocket), static classifiers (SVM/KNN/DT/RF/XGBoost) |
| **SEED & DEAP** | `SourceFiles/SEED&DEAP/` | Shared library (`lib/`) + per-dataset baseline scripts (EEGNet, GRU, LSTM) using CD-FES vs. raw/sub-band EEG |
| **Self-Collected Depression** | `SourceFiles/SelfCollectedDepression/` | Proposed two-level fusion LSTM, feature comparisons (CD-FES, CD-AED, DE), ablations (band-wise and mode-wise), and 8 replication methods (EEGNet, ATCNet, EEGConformer, EEGMiner, EEGNeX, SFCSAN, Shen 2017, TiSc) |

### ExperimentLogs

Mirror the SourceFiles structure. Each leaf directory contains `.log` files and `.tar.gz` archives of saved TensorBoard logs and model weights.

| Dataset | Log highlights |
|---------|---------------|
| **DEAP** | Arousal / Valence × FES vs. rawSubBand × 3 models |
| **SEED** | Sessions 1–3 × FES vs. OriginalSubBand × 3 models, plus summary F1 comparison plots |
| **Depression** | Proposed method logs, feature comparisons, ablations (band/mode), 8 replication methods |
| **MODMA** | Preprocessing log, static classifier tuning, feature-by-one results, dynamic modeling (LSTM/GRU), MiniRocket |

## Key methods

- **SPWVD** — Smoothed Pseudo Wigner-Ville Distribution for time–frequency analysis
- **FES** — Frequency-domain Entropy Sequence, a dimensionless entropy measure derived from normalised log-spectra
- **CD-FES** — Common- and differential-mode FES from FP1/FPz/FP2 channels
- **Two-level fusion LSTM** — Per-band Stream LSTM + Squeeze-and-Excitation band fusion + DenseRes mode fusion
