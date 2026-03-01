Before execute main files, do ```source ./env.sh``` first. This will add ```lib``` package into python's package list temporarily.

The main files are configured for FES validation. To run the baseline with raw EEG sub-frequency bands, substitute ```lib.EEG.preprocess_as_FES``` with ```lib.EEG.no_preprocess```. Function ```lib.EEG.preprocess_as_FES``` defines the definition of the proposed FES (in simplified version).
