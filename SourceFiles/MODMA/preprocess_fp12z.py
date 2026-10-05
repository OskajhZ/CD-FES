"""
MODMA 128-channel resting-state EEG → FP1/FPz/FP2 preprocessing pipeline

Pipeline:
  1. Load 53 subjects' .mat files + subjects_information.xlsx
  2. Detrend
  3. 50 Hz notch (power-line interference removal)
  4. Bandpass filter (0.5-45 Hz)
  5. ICA ocular artifact removal
  6. Extract FP1, FPz, FP2 channels
  7. Incremental per-subject HDF5 save

Channel map (ref: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_eeg/EEG_128_channel_array_map.pdf):
  FP2 → EGI#9  → row 8
  FPz → EGI#15 → row 14
  FP1 → EGI#22 → row 21

Output HDF5 structure:
  data:   (n_subjects, 3, n_timepoints)  float64  (appended incrementally)
  labels: (n_subjects,)                   float64  (0=HC, 1=MDD)
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import glob
import os
import h5py
import mne
from scipy import signal as scipy_signal
from mne.preprocessing import ICA


# ============================================================
# 0. Paths
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(CURRENT_DIR)
XLSX_PATH = os.path.join(DATA_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "eeg_fp123_cleaned.h5")


# ============================================================
# 1. Load subject info table
# ============================================================
def load_subject_info(xlsx_path):
    info = pd.read_excel(xlsx_path)
    info = info.dropna(how="all", axis=1)
    info["file_prefix"] = info["subject id"].apply(lambda x: f"{x:07d}")
    info["group_label"] = info["type"].map({"MDD": 1.0, "HC": 0.0})
    return info


# ============================================================
# 2. Load a single subject's .mat file
# ============================================================
def load_subject_eeg(mat_dir, file_prefix):
    """Return: (eeg_data: (128, T), sampling_rate, impedances)"""
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    matches = [f for f in mat_files if file_prefix in os.path.basename(f)]
    if not matches:
        raise FileNotFoundError(f"No .mat file matching {file_prefix}")

    data = sio.loadmat(matches[0], squeeze_me=True)
    keys = [k for k in data.keys()
            if not k.startswith("__")
            and k != "samplingRate"
            and "Impedances" not in k]
    eeg = data[keys[0]][:128, :]

    sr = float(data["samplingRate"])
    imp_key = [k for k in data.keys() if "Impedances" in k]
    imp = data[imp_key[0]] if imp_key else None
    return eeg, sr, imp


# ============================================================
# 3. ICA artifact removal
# ============================================================
def clean_eeg_with_ica(eeg, sfreq=250.0,
                       lowcut=0.5, highcut=50.0,
                       notch=50.0,
                       eog_ch_indices=(24, 20, 13, 7),
                       random_state=42):
    """
    Detrend → notch → bandpass → ICA → rebuild.

    EOG detection uses a 4.5 Hz low-pass on reference channels to avoid
    frontal alpha/theta being mistaken for ocular artifacts.
    """
    ch_names = [f"E{i+1}" for i in range(128)]
    ch_types = ["eeg"] * 128
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    eeg_detrended = scipy_signal.detrend(eeg, axis=-1, type="linear")
    raw = mne.io.RawArray(eeg_detrended, info, verbose="ERROR")
    raw.notch_filter(notch, fir_design="firwin", verbose="ERROR")
    raw.filter(lowcut, highcut, fir_design="firwin", verbose="ERROR")

    ica = ICA(n_components=None, method="fastica",
              random_state=random_state, max_iter=500,
              fit_params=dict(tol=1e-3))
    ica.fit(raw, verbose="ERROR")

    raw_eog = raw.copy()
    raw_eog.filter(None, 4.5, fir_design="firwin", verbose="ERROR")
    eog_idx, eog_scores = ica.find_bads_eog(
        raw_eog, ch_name=[f"E{i+1}" for i in eog_ch_indices],
        threshold=4, verbose="ERROR")
    if len(eog_idx) == 0:
        eog_idx, eog_scores = ica.find_bads_eog(
            raw_eog, ch_name=[f"E{i+1}" for i in eog_ch_indices],
            threshold=3.5, verbose="ERROR")

    ica.exclude = eog_idx
    cleaned = ica.apply(raw, verbose="ERROR").get_data()
    return cleaned, len(eog_idx)


# ============================================================
# 4. Incremental HDF5 read/write
# ============================================================
def append_to_hdf5(h5_path, data_triple, label):
    """
    Append one subject, truncating all to the minimum length.
    """
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "a") as f:
            old_n = f["data"].shape[0]
            old_T = f["data"].shape[2]
            new_T = data_triple.shape[1]
            T_keep = old_T if old_T < new_T else new_T

            new_data = np.zeros((old_n + 1, 3, T_keep), dtype=np.float64)
            new_data[:old_n] = f["data"][:, :, :T_keep]
            new_data[-1] = data_triple[:, :T_keep]
            del f["data"]
            f.create_dataset("data", data=new_data, dtype="float64")

            new_labels = np.zeros(old_n + 1, dtype=np.float64)
            new_labels[:old_n] = f["labels"][:]
            new_labels[-1] = label
            del f["labels"]
            f.create_dataset("labels", data=new_labels, dtype="float64")
    else:
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data",
                             data=data_triple[np.newaxis, :, :],
                             dtype="float64",
                             maxshape=(None, 3, None))
            f.create_dataset("labels",
                             data=np.array([label], dtype=np.float64),
                             dtype="float64",
                             maxshape=(None,))


# ============================================================
# 5. Visualisation
# ============================================================
def visualize_subject(raw_eeg, cleaned_eeg, sr, subject_id, n_eog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for font_name in ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
                       "SimHei", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    fig, axes = plt.subplots(3, 2, figsize=(14, 6))
    ch_labels = ["FP1", "FPz", "FP2"]
    rows = [21, 14, 8]

    n_samples = int(sr * 2)
    n_total = raw_eeg.shape[1]
    offset = n_total // 3
    time = np.arange(n_samples) / sr

    for i, (ch_name, row_idx) in enumerate(zip(ch_labels, rows)):
        ax1 = axes[i, 0]
        ax1.plot(time, raw_eeg[row_idx, offset:offset+n_samples], "r-", alpha=0.7, linewidth=0.5)
        ax1.set_ylabel("\u00b5V")
        ax1.set_title(f"Raw (filtered)", fontsize=10)
        ax1.set_xlabel("Time (s)")

        ax2 = axes[i, 1]
        ax2.plot(time, cleaned_eeg[row_idx, offset:offset+n_samples], "b-", alpha=0.7, linewidth=0.5)
        ax2.set_title(f"ICA cleaned", fontsize=10)
        ax2.set_xlabel("Time (s)")

    fig.suptitle(f"Subject {subject_id}  |  EOG removed:{n_eog}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(DATA_DIR, "qc_figures")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{subject_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    QC figure saved: {path}")
    return path


# ============================================================
# 6. Main
# ============================================================
def main():
    print("=" * 60)
    print("MODMA 128ch EEG -> FP1/FPz/FP2 preprocessing (incremental save)")
    print("=" * 60)

    info = load_subject_info(XLSX_PATH)
    print(f"\nTotal {len(info)} subjects: MDD={(info['type']=='MDD').sum()}, "
          f"HC={(info['type']=='HC').sum()}")

    total = len(info)

    for idx, (_, row) in enumerate(info.iterrows()):
        prefix = row["file_prefix"]
        label = row["group_label"]
        print(f"\n[{idx+1}/{total}] {prefix} {'MDD' if label==1 else 'HC'}  ", end="", flush=True)

        try:
            eeg_raw, sr, _ = load_subject_eeg(DATA_DIR, prefix)
            assert sr == 250.0, f"Sampling rate {sr} != 250"
            print(f"Loaded {eeg_raw.shape[1]} samples", end=" | ", flush=True)

            eeg_clean, n_eog = clean_eeg_with_ica(eeg_raw, sfreq=sr)
            print(f"ICA: EOG={n_eog}", end=" | ", flush=True)

            triple = np.stack([
                eeg_clean[21, :],
                eeg_clean[14, :],
                eeg_clean[8,  :],
            ], axis=0)

            append_to_hdf5(OUTPUT_PATH, triple, label)
            print(f"Saved (3x{triple.shape[1]})", end="", flush=True)

            if True:
                eeg_pre = scipy_signal.detrend(eeg_raw, axis=-1, type="linear")
                ch_names = [f"E{i+1}" for i in range(128)]
                ch_types = ["eeg"] * 128
                info_mne = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types=ch_types)
                raw_pre = mne.io.RawArray(eeg_pre, info_mne, verbose="ERROR")
                raw_pre.notch_filter(50, fir_design="firwin", verbose="ERROR")
                raw_pre.filter(0.5, 45.0, fir_design="firwin", verbose="ERROR")
                raw_data = raw_pre.get_data()

                fig_path = visualize_subject(
                    raw_data, eeg_clean, sr, prefix, n_eog)
                print(f"  QC saved", end="", flush=True)

            print()

        except Exception as e:
            print(f"\n  !! Error: {e}")
            import traceback
            traceback.print_exc()

    if os.path.exists(OUTPUT_PATH):
        with h5py.File(OUTPUT_PATH, "r") as f:
            n = f["data"].shape[0]
            t = f["data"].shape[2]
            mdd = (f["labels"][:] == 1).sum()
            hc = (f["labels"][:] == 0).sum()
        print(f"\n{'='*60}")
        print(f"Done!")
        print(f"  HDF5: {OUTPUT_PATH}")
        print(f"  data:   ({n}, 3, {t})  float64")
        print(f"  labels: MDD={mdd}, HC={hc}")
        print(f"{'='*60}")
    else:
        print(f"\n!! No output generated")


if __name__ == "__main__":
    main()
