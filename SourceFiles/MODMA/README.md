# MODMA Depression Recognition (CD-FES)

Code accompanying _Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition_.

## Pipeline

```
raw 128ch EEG  →  preprocess_fp12z.py  →  FP1/FPz/FP2 (HDF5)
                                ↓
                    feature_define.py (SPWVD → AED/FES)
                                ↓
                    CD-FES (differential + common mode)
                                ↓
         ┌───────────────────────────────────────┐
         │   StaticClassifier/                   │
         │   ├── main.py         — proposed LSTM │
         │   │                     + ML baselines │
         │   └── reproduction.py — Shen 2017 &   │
         │                         Cai 2018      │
         └───────────────────────────────────────┘
         ┌───────────────────────────────────────┐
         │   DynamicModeling/                    │
         │   ├── main.py          — LSTM/GRU     │
         │   │                     baselines     │
         │   └── minirocket_main.py — MiniRocket │
         │                          + Ridge      │
         └───────────────────────────────────────┘
```

## Files

| File | Role |
|---|---|
| `preprocess_fp12z.py` | 128ch → FP1/FPz/FP2: detrend, 50 Hz notch, 0.5–45 Hz bandpass, ICA (EOG removal), incremental HDF5 dump |
| `StaticClassifier/feature_define.py` | SPWVD time–frequency analysis; CD-AED / CD-FES extraction; 10-fold cross-validation iterator |
| `StaticClassifier/model_define.py` | Two-level fusion LSTM, Stream + SE feature extractor, DenseRes, trainers, ensemble voting |
| `StaticClassifier/main.py` | **Proposed method:** CD-FES + two-level fusion LSTM; ML moments baselines (SVM/KNN/DT/RF/XGBoost) |
| `StaticClassifier/reproduction.py` | Reproduction of Shen et al. 2017 and Cai et al. 2018 on the same MODMA data |
| `DynamicModeling/feature_define.py` | Same SPWVD pipeline; includes `RawEEG_CrossValidationIter` for raw/subband EEG |
| `DynamicModeling/model_define.py` | `BaselineDynamicModel` (LSTM/GRU), same trainer/ensemble/metrics as static branch |
| `DynamicModeling/main.py` | LSTM & GRU baselines on EEG / subband / CD-AED / CD-FES |
| `DynamicModeling/minirocket_main.py` | MiniRocket (10k kernels) + RidgeClassifierCV baseline |

## Requirements

`pip install numpy scipy pandas h5py mne torch scikit-learn xgboost sktime nolds matplotlib seaborn tqdm`

## Usage

```bash
# Preprocessing
python preprocess_fp12z.py

# StaticClassifier: proposed LSTM + ML baselines
cd StaticClassifier && python main.py

# StaticClassifier: reproduce other static features
cd StaticClassifier && python reproduction.py

# DynamicModeling: LSTM/GRU baselines
cd DynamicModeling && python main.py

# DynamicModeling: MiniRocket baseline
cd DynamicModeling && python minirocket_main.py
```

Adjust the hard-coded `MODMA_SPECTRUM_DIR` / `base_dir` paths in each script before running.
