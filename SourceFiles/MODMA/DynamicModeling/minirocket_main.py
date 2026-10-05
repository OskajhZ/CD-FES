'''
Author:
    Xiangnan Zhang: zhangxn@bit.edu.cn
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025

MiniRocket + RidgeClassifierCV baseline for non-neural time series
classification on CD-FES. Reuses the same data pipeline as Table III
(LSTM/GRU baselines) for strict comparability.

Reference:
    A. Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic
    Transform for Time Series Classification," KDD 2021.

The code is under the article: Frequency-Domain Entropy Sequence as a
Dynamic EEG Representation for Depression Recognition.
'''



import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import feature_define
import model_define

from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from sktime.transformations.panel.rocket import MiniRocket


np.random.seed(42)

RESULTS_FILE = "minirocket_results.json"


def minirocket_cross_verification(spectrum_dir: str, data_type, **kwargs):
    ensemble_amount = kwargs.get("ensemble_amount", 100)
    fold_amount = kwargs.get("fold_amount", 10)
    ridge_alpha = kwargs.get("ridge_alpha", 1.0)

    assert data_type in ["CDFES", "CDAED", "FES", "AED", "EEG", "subband"], \
        f"Invalid data_type: {data_type}"

    accuracy_list = []
    record_list = []
    confusion_matrix_list = []

    start_fold = 0
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            saved = json.load(f)
        if saved.get("data_type") == data_type and saved.get("fold_amount") == fold_amount:
            accuracy_list = saved["accuracy_list"]
            record_list = saved["record_list"]
            confusion_matrix_list = saved["confusion_matrix_list"]
            start_fold = len(accuracy_list)
            print(f"Resuming from fold {start_fold + 1}")

    folds = None
    if data_type in ["AED", "FES", "CDAED", "CDFES"]:
        folds = feature_define.SpectrumCrossValidationIter(spectrum_dir, fold_amount, data_type)
    elif data_type == "EEG":
        folds = feature_define.RawEEG_CrossValidationIter(spectrum_dir, fold_amount, subband=False)
    elif data_type == "subband":
        folds = feature_define.RawEEG_CrossValidationIter(spectrum_dir, fold_amount, subband=True)

    for i, (train_set, validate_set) in enumerate(folds):
        if i < start_fold:
            continue

        fold_idx = i + 1
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}/{fold_amount}")
        print(f"{'='*50}")

        X_train = train_set["data"]
        y_train = train_set["labels"]
        X_test = validate_set["data"]
        y_test = validate_set["labels"]
        good_list = validate_set["good_list"]

        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  Channels: {X_train.shape[1]}, Time points: {X_train.shape[-1]}")

        t0 = __import__("time").time()
        mini = MiniRocket(num_kernels=10000, max_dilations_per_kernel=4,
                          n_jobs=-1, random_state=42)
        mini.fit(X_train)
        X_train_t = mini.transform(X_train)
        X_test_t = mini.transform(X_test)
        t1 = __import__("time").time()
        print(f"  MiniRocket transform: {t1-t0:.1f}s, features: {X_train_t.shape[1]}")

        clf = RidgeClassifier(alpha=ridge_alpha)
        clf.fit(X_train_t, y_train)
        t2 = __import__("time").time()
        print(f"  Ridge fit (alpha={ridge_alpha}): {t2-t1:.1f}s")

        clip_preds = clf.predict(X_test_t)
        ensemble_pred, ensemble_labels = model_define.make_ensemble(
            clip_preds, y_test, ensemble_amount, good_list, "category")
        ensemble_result = model_define.make_statistics(ensemble_pred, ensemble_labels)
        confusion_matrix = metrics.confusion_matrix(ensemble_labels, ensemble_pred)

        accuracy_list.append(ensemble_result["accuracy"])
        record_list.append(ensemble_result)
        confusion_matrix_list.append(confusion_matrix.tolist())

        print(f"  Fold {fold_idx} accuracy: {ensemble_result['accuracy']:.4f}, "
              f"F1: {ensemble_result['f1']:.4f}, "
              f"precision: {ensemble_result['precision']:.4f}, "
              f"recall: {ensemble_result['recall']:.4f}")
        print(f"  Running avg ({len(accuracy_list)}/{fold_amount}): "
              f"{np.mean(accuracy_list):.4f}")

        with open(RESULTS_FILE, "w") as f:
            json.dump({
                "data_type": data_type,
                "fold_amount": fold_amount,
                "accuracy_list": accuracy_list,
                "record_list": record_list,
                "confusion_matrix_list": confusion_matrix_list,
            }, f, indent=2)
        sys.stdout.flush()

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    return np.array(accuracy_list), record_list, confusion_matrix_list


def main():
    spectrum_dir = "/home/xiangnan/E/EDoc/Research/Dataset/MODMA/EEG_128channels_resting_lanzhou_2015/preprocessed/spectrum.h5"

    data_type_list = ["CDFES", "CDAED", "subband", "EEG"]

    for data_type in data_type_list:
        print(f"\n{'='*60}")
        print(f"MiniRocket + Ridge on {data_type}")
        print(f"{'='*60}")
        sys.stdout.flush()

        kwargs = {"ensemble_amount": 100, "fold_amount": 10, "ridge_alpha": 5000.0}

        accuracy_arr, record_list, confusion_matrix_list = minirocket_cross_verification(
            spectrum_dir, data_type, **kwargs)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: {data_type}")
        print(f"{'='*60}")

        n_folds = len(accuracy_arr)
        prec_arr = np.array([r["precision"] for r in record_list])
        rec_arr  = np.array([r["recall"] for r in record_list])
        f1_arr   = np.array([r["f1"] for r in record_list])

        def fmt_mean_std_err(arr):
            return f"{arr.mean()*100:.2f}$\\pm${arr.std()/np.sqrt(n_folds)*100:.2f}"

        print(f"  Accuracy:  {fmt_mean_std_err(accuracy_arr)} %")
        print(f"  Precision: {fmt_mean_std_err(prec_arr)} %")
        print(f"  Recall:    {fmt_mean_std_err(rec_arr)} %")
        print(f"  F1:        {fmt_mean_std_err(f1_arr)} %")


if __name__ == "__main__":
    main()
