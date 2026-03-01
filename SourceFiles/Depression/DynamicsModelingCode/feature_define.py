'''
Author: 
    Xiangnan Zhang: zhangxn@bit.edu.cn 
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''



import numpy as np
import scipy
import h5py
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from typing import *
import multiprocessing
import os
from tqdm import tqdm

plt.rcParams['axes.unicode_minus'] = False

frequency_ranges = [(4,8), (8, 14), (14, 31), (31, 40)]
band_names = ["Theta", "Alpha", "Beta", "Gamma"]

# Follows: convert EEG signals to prower spectrum series

def spwvd(real_signal: np.ndarray, freq_range: Tuple[int, int] = None, fs = 125) -> np.ndarray:
    '''
    ESSENTIAL COMPONENT!!! FUNDAMENTAL!!! CORNERSTONE!!!
    Input: real_signal (length near 384)
    Output: TFR[frequency, time]
        if freq_range != None, trim TFR into referred frequency range (Hz)
    '''
    # Configuration
    length = len(real_signal)
    max_delay = length//4
    tail = np.zeros(max_delay)
    real_signal = np.concatenate([tail, real_signal, tail], axis=0)
    zero_idx = len(tail)
    total_delay = 2 * max_delay + 1 # Positive delay + negative + zero
    t_window = scipy.signal.windows.hann(61)
    t_window /= np.sum(t_window)
    f_window = scipy.signal.windows.hann(total_delay)
    f_window /= np.sum(f_window)

    # Step 1: convert to analytic signal
    analytic_signal = scipy.signal.hilbert(real_signal)

    # Step 2: compute instantaneous autocorrelation matrix K[t, tau]
    K = np.zeros((length, total_delay), dtype = np.complex128)
    t = np.arange(0, length)
    for tau in range(-max_delay, max_delay+1): # Use conjugate symmetry of K
        K[t, max_delay+tau] = analytic_signal[zero_idx+t+tau//2] * np.conj(analytic_signal[zero_idx+t-tau//2])

    # Step 3: smooth along time
    R = np.zeros_like(K)
    for tau in range(total_delay):
        R[:, tau] = scipy.signal.convolve(K[:, tau], t_window, mode="same", method="auto")

    # Step 4: smooth along frequency, get TFR
    SPWVD = np.zeros((length, length), dtype=np.complex128) # store the result of FFT, so the dtype is not real
    for t in range(length):
        row = R[t, :] * f_window
        SPWVD[t, :] = np.fft.fftshift(np.fft.fft(row, n=length))
    TFR = SPWVD.T

    if freq_range is not None:
        start_freq, end_freq = freq_range
        zero_freq = len(TFR)//2
        sampling_time = length // fs
        TFR = TFR[zero_freq+start_freq*sampling_time : zero_freq+end_freq*sampling_time, :]

    return TFR # complex type for elements


class BandpassFilter():
    def __init__(self, base_signal, fs, order):
        self.base_signal = base_signal
        self.frequency = fs
        self.order = order
    def filtrate(self, lowcut, highcut):
        nyq = 0.5 * self.frequency
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(self.order, [low, high], btype="band")
        y = scipy.signal.lfilter(b, a, self.base_signal)
        return y


def filtrate_a_signal(signal):
    '''
    signal: [len] for 250 Hz
    Return: [4, 1, len//2], downsample to 125 Hz
    '''
    filtrated_signals = []
    ffilter = BandpassFilter(signal, 250, 6)
    for lowcut, highcut in frequency_ranges:
        filtrated_signals.append(ffilter.filtrate(lowcut, highcut)[np.newaxis, np.newaxis, ::2]) # downsample after filtrating
    filtrated_signals = np.vstack(filtrated_signals)
    return filtrated_signals


def clip_decompose(clip):
    '''
    band: [4, 3, time*125] of 125 Hz
    Return: [1, 4, 3, time*125]
    '''
    energy = []
    for band_idx, singleband in enumerate(clip):
        TFRs = []
        freq_begin, freq_end = frequency_ranges[band_idx]
        for singlechannel in singleband:
            TFR = spwvd(singlechannel, (freq_begin, freq_end), 125) # mean along frequency
            TFR = np.abs(TFR)
            TFR = TFR.mean(-2)[np.newaxis]
            TFRs.append(TFR)
        TFRs = np.vstack(TFRs)[np.newaxis]
        energy.append(TFRs)
    energy = np.vstack(energy)[np.newaxis]
    return energy


def extract_CDAED(EEG_dir: str, save_dir: str) -> Tuple[tuple, tuple]:
    '''
    Main function to convert EEG signals to spectrum series
    After convertion:
        Features in dataset: [subjects, clips, 4 bands, 3 channels, 125]
            1s for each clip.
            3 channels for Fp1, Fpz, Fp2
        Labels in dataset: [subjects, clips]
    '''

    process_pool = multiprocessing.Pool()
    window = 3*125

    def decompose_filtrated_signals(signal):
        '''
        Segment full-length filtrated EEG sigals, Return spectrum clips with equal length as window
        signal: [4, 3, length], 4 means 4 bands
        Return: [length//window, 4, 3, 3*125]
        '''
        clip_num = signal.shape[-1] // window
        band_amount = signal.shape[0]
        channel_amount = signal.shape[1]

        EEG_clips = np.transpose(signal, (2, 0, 1)) # [length, 4, 3]
        EEG_clips = EEG_clips.reshape(clip_num, window, band_amount, channel_amount)
        EEG_clips = np.transpose(EEG_clips, (0, 2, 3, 1)) # [clip_num, 4, 3, window]

        spectrums = process_pool.map(clip_decompose, EEG_clips)

        spectrums = np.vstack(spectrums)

        return spectrums, EEG_clips

    def preview(filtrated_signal): # To verify the effect of SPWVD
        channel = np.random.randint(0, filtrated_signal.shape[1])
        start_idx = np.random.randint(0, filtrated_signal.shape[-1] - (window))
        clip_signal = filtrated_signal[:, channel, start_idx: start_idx+window] # randomly select a 3s clip
        TFRs = []
        for band in clip_signal:
            TFR = spwvd(band, (0, 50), 125)[np.newaxis]
            TFRs.append(np.abs(TFR))
        TFRs = np.vstack(TFRs)
        total_TFR = TFRs.sum(0)
        fig = plt.figure(figsize=(28, 8))
        plt.subplot(2,1,1)
        plt.plot(clip_signal.sum(0))
        plt.title("Filtrated Signal")
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.subplot(2,1,2)
        plt.title("SPWVD Preview")
        plt.imshow(total_TFR)
        plt.ylabel("Frequency domain")
        plt.xlabel("Time domain")
        plt.tight_layout()
        plt.show()
        fig.savefig("SPWVD_preview.jpg")

    with h5py.File(EEG_dir, "r") as root:
        EEG = root["data"][:] # [subjects, channels, length]
        EEG_labels = root["labels"][:] # [subjects, ]
        EEG_dataset = zip(EEG, EEG_labels)
        _, channels, _ = EEG.shape

    spectrum_set = {"data": [], "labels": [], "EEG_clips": []}
    selected_indices = []

    for idx, (subject, label) in tqdm(enumerate(EEG_dataset), total=len(EEG_labels)):
        filtrated_signal_list = process_pool.map(filtrate_a_signal, subject)
        filtrated_signal = np.concatenate(filtrated_signal_list, axis=1) # [4, 3, length]
        if idx==0:
            preview(filtrated_signal)
        spectrums, EEG_clips = decompose_filtrated_signals(filtrated_signal)
        spectrums = spectrums[np.newaxis]
        EEG_clips = EEG_clips[np.newaxis]
        if spectrums.mean() > 1e2:
            print("Bad Subject. Skip.")
        else:
            selected_indices.append(idx)
            spectrum_set["data"].append(spectrums)
            spectrum_set["EEG_clips"].append(EEG_clips)
            expanded_labels = np.repeat(label, spectrums.shape[1])[np.newaxis]
            spectrum_set["labels"].append(expanded_labels)
    
    spectrum_data = np.vstack(spectrum_set["data"]) # [subjects, clips, 4, 3, window]
    spectrum_EEG_clips = np.vstack(spectrum_set["EEG_clips"])
    spectrum_labels = np.vstack(spectrum_set["labels"]) # [subjects, clips]
    EEG = EEG[selected_indices]
    EEG_labels = EEG_labels[selected_indices]
    
    with h5py.File(save_dir, "w") as root:
        root.create_dataset("data", data=spectrum_data)
        root.create_dataset("labels", data=spectrum_labels)
        root.create_dataset("EEG_clips", data=spectrum_EEG_clips)
        EEG_group = root.create_group("OriginalEEG")
        EEG_group.create_dataset("data", data=EEG)
        EEG_group.create_dataset("labels", data=EEG_labels)

    return spectrum_data.shape, spectrum_labels.shape, EEG.shape, EEG_labels.shape

# Follows: convert spectrum series to entropy seires

def unpack(x, y, select=False, ref_spectrum=None):
    if ref_spectrum is None:
        ref_spectrum = x
    x = x.reshape(-1, *x.shape[2:])
    ref_spectrum = ref_spectrum.reshape(-1, *ref_spectrum.shape[2:])
    y = y.reshape(-1)
    good_list = ref_spectrum.mean(-1).mean(-1).mean(-1)<=1e2 
    if select:
        x = x[good_list]
        y = y[good_list]
        retain_ratio = good_list.sum() / len(good_list)
        return x, y
    else:
        return x, y, good_list


def convert_to_FES_series(train_x, validate_x):
    '''
    x: [num, bands, channels, length]
    '''
    unit_coef = train_x.mean(-1, keepdims=True).mean(0) # Normalize a whole band with single coefficient
    print("unit_coef:")
    print(unit_coef.mean(-1))
    train_x = np.log(train_x/unit_coef) + 1
    validate_x = np.log(validate_x/unit_coef) + 1
    return train_x, validate_x, unit_coef


def get_diff_and_com(x):
    '''
    Get differential and common mode
    '''
    Fpz = x[..., 1, :][..., np.newaxis, :]
    diff = x[..., 0, :] - x[..., 2, :]
    diff = diff[..., np.newaxis, :]
    new_x = np.concatenate([diff, Fpz], axis = -2)
    return new_x


def standarize(train_x, validate_x):
    '''
    x: [num, bands, channels, length]
    '''
    reduced_data = train_x.mean(-1)
    means = reduced_data.mean(0)[..., np.newaxis]
    stds = reduced_data.std(0)[..., np.newaxis]
    _, bands, channels, length = train_x.shape
    means = np.tile(means, (1, 1, length))
    stds = np.tile(stds, (1, 1, length))
    train_x = (train_x-means)/stds
    validate_x = (validate_x-means)/stds
    return train_x, validate_x


def process_spectrum(train_x, validate_x, to_FES=True, to_CD=True):
    '''
    In: unpacked spectrums
    Out: final entropy series
    '''
    if to_FES:
        train_x, validate_x, _ = convert_to_FES_series(train_x, validate_x)
    if to_CD:
        train_x = get_diff_and_com(train_x)
        validate_x = get_diff_and_com(validate_x)
    train_x, validate_x = standarize(train_x, validate_x)
    return train_x, validate_x


def visualize_entropy(dataset: dict) -> matplotlib.figure.Figure:
    '''
    dataset = {"data": np.ndarray, "labels": np.ndarray}
        data: [subjects, clips, 4 bands, 2 channels, len], assume len = 125
        labels: [subjects, clips]
    '''
    # Prepare data
    sample_list = []
    seek_label = 0
    for clip_idx, data in enumerate(dataset["data"]):
        if dataset["labels"][clip_idx] == seek_label:
            sample_list.append(data)
            seek_label += 1
            if seek_label > 1:
                break

    categories = ["Normal", "Depression"]
    channel_labels = ["Differential Mode", "Common Mode"]
    band_names = ["Theta", "Alpha", "Beta", "Gamma"]
    sample_with_category = list(zip(sample_list, categories))

    # Set plot style and larger font
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
#        "axes.titleweight": "bold",
#        "axes.labelweight": "bold",
        "figure.dpi": 300
    })

    figure = plt.figure(figsize=(15, 15))
    channels = 2
    bands = 4

    line_styles = {
            "Normal": {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2},
            "Depression": {"color": "#d62728", "linestyle": "--", "linewidth": 2}
        }

    for band_idx in range(bands):
        for channel_idx in range(channels):
            ax = plt.subplot(4, 2, band_idx * channels + channel_idx + 1)
            for i, (sample, category) in enumerate(sample_with_category):
                ax.plot(sample[band_idx, channel_idx],
                        label=category,
                        **line_styles[category])
            # Axis formatting
            if band_idx == 0:
                ax.set_title(channel_labels[channel_idx])
            if channel_idx == 0:
                ax.set_ylabel(band_names[band_idx])
            if band_idx == bands - 1:
                ax.set_xlabel("Time")
            ax.grid(True, linestyle='--', alpha=0.5)

#            if band_idx == 0 and channel_idx == 1:
            ax.legend(loc="upper right", frameon=True, framealpha=0.8)

    plt.tight_layout()

    return figure

def global_CDFES_computing(EEG_dir) -> Tuple[np.ndarray]:
    with h5py.File(spectrum_dir, "r") as root:
        spectrum = root["data"][:]
        labels = root["labels"][:]
    spectrum, labels = unpack(spectrum, labels, True)
    FES, _, _ = convert_to_FES_series(spectrum, spectrum)
    CDFES = get_diff_and_com(FES)
    return CDFES, labels

class SpectrumCrossValidationIter():

    def __init__(self, dataset_dir: str, fold_num = 10, data_type="CDFES"):
        '''
        data_type: "FES", "AED", "CDFES", "CDAED"
        '''

        if data_type!="FES" and data_type!="AED" and data_type!="CDFES" and data_type!="CDAED":
            raise ValueError("Invalid data_type: {}".format(data_type))

        np.random.seed(42)
        self.total_ref_spectrums = None
        self.total_data = None
        self.total_labels = None
        with h5py.File(dataset_dir, "r") as root:
            self.total_data = root["data"][:] # [subjects, clips, 4 bands, 3 channels, 125]
            self.total_labels = root["labels"][:]
            if data_type=="CDEEG":
                self.total_data = root["EEG_clips"][:]
                self.total_ref_spectrums = root["data"][:]
            else:
                self.total_data = root["data"][:]
                self.total_ref_spectrums = self.total_data

        shuffle_list = np.arange(len(self.total_data))
        np.random.shuffle(shuffle_list)
        self.total_data = self.total_data[shuffle_list]
        self.total_labels = self.total_labels[shuffle_list]
        self.total_ref_spectrums = self.total_ref_spectrums[shuffle_list]

        self.fold = 0
        self.fold_num = fold_num
        self.fold_scale = len(self.total_data) // self.fold_num

        self.data_type = data_type
    
    def __iter__(self):
        self.fold = 0
        return self

    def clip(self, start_idx, end_idx):
        train_set = {}
        validate_set = {}
        validate_indices = np.arange(start_idx, end_idx)
        validate_set["data"] = self.total_data[validate_indices]
        validate_set["labels"] = self.total_labels[validate_indices]
        validate_ref_spectrums = self.total_ref_spectrums[validate_indices]
        train_set["data"] = np.delete(self.total_data, validate_indices, axis=0)
        train_set["labels"] = np.delete(self.total_labels, validate_indices, axis=0)
        train_ref_spectrums = np.delete(self.total_ref_spectrums, validate_indices, axis=0)

        validate_set["data"], validate_set["labels"], validate_set["good_list"] = unpack(
                validate_set["data"], validate_set["labels"], False, validate_ref_spectrums
                )
        train_set["data"], train_set["labels"] = unpack(
                train_set["data"], train_set["labels"], True, train_ref_spectrums
                )
        to_FES = True if self.data_type in ["FES", "CDFES"] else False
        to_CD = True if self.data_type in ["CDFES", "CDAED"] else False
        train_set["data"], validate_set["data"] = process_spectrum(train_set["data"], validate_set["data"], to_FES, to_CD)
        train_set["data"] = train_set["data"].reshape(train_set["data"].shape[0], -1, train_set["data"].shape[-1])
        validate_set["data"] = validate_set["data"].reshape(validate_set["data"].shape[0], -1, validate_set["data"].shape[-1])
        return train_set, validate_set

    def __next__(self):
        if self.fold >= self.fold_num:
            raise StopIteration
        else:
            train_set: dict = None
            validate_set: dict = None
            if self.fold == 0 and self.data_type == "CDFES":
                train_set, validate_set = self.clip(0, self.fold_scale)
                visualize_figure = visualize_entropy(validate_set)
                visualize_figure.savefig("CDFES_visualize.pdf")
            elif self.fold == self.fold_num-1:
                train_set, validate_set = self.clip(self.fold*self.fold_scale, len(self.total_data))
            else:
                train_set, validate_set = self.clip(self.fold*self.fold_scale, (self.fold+1)*self.fold_scale)
            self.fold += 1
            return train_set, validate_set


def convert_series_to_moments(CDFES: np.ndarray) -> np.ndarray:
    mean = np.mean(CDFES, axis=-1, keepdims=True)
    std = np.std(CDFES, axis=-1, ddof=1, keepdims=True)
#    skew = scipy.stats.skew(CDFES, axis=-1, bias=False, keepdims=True)
#    kurtosis = scipy.stats.kurtosis(CDFES, axis=-1, fisher=True, bias=False, keepdims=True)
    moments = np.concatenate([mean, std], axis=-1)
    moments = moments.reshape(*moments.shape[:-3], -1)
    return moments

def chunk_and_expand(EEG, labels, clip_amount=28):
    '''
    [subjects, channels, length]
    '''
    subjects, channels, length = EEG.shape
    window = length // clip_amount
    EEG = np.transpose(EEG, (0, 2, 1))
    EEG = EEG.reshape(subjects, clip_amount, window, channels)
    EEG = np.transpose(EEG, (0, 1, 3, 2)) # [sj, clip, channel, length]
    EEG = EEG.reshape(subjects*clip_amount, channels, window)
    labels = np.repeat(labels, clip_amount)
    return EEG, labels

def arrange(EEG, labels, ref_spectrum, select=True):
    '''
    '''
    EEG, labels = chunk_and_expand(EEG, labels, 28)
    ref_spectrum = ref_spectrum.reshape(-1, *ref_spectrum.shape[2:])
    good_list = ref_spectrum.mean(-1).mean(-1).mean(-1)<=1e2
    if select:
        EEG_selected = EEG[good_list]
        labels_selected = labels[good_list]
        return EEG_selected, labels_selected
    else:
        return EEG, labels, good_list

def filtrate_all_EEGs(EEGs):
    '''
    EEGs: [subject, channel, time]
    return: [subject, channel*4, time//2]
    '''
    subject, channel, length = EEGs.shape
    downsampled_length = length // 2
    filtrated = np.zeros([subject, channel*4, downsampled_length])
    for subject_idx in range(subject):
        for channel_idx in range(channel):
            filtrated_channel_start = 4*channel_idx
            filtrated_channel_end = filtrated_channel_start + 4
            single = filtrate_a_signal(EEGs[subject_idx, channel_idx])
            single = single.reshape(4, downsampled_length)
            filtrated[subject_idx, filtrated_channel_start:filtrated_channel_end] = single
    return filtrated

class RawEEG_CrossValidationIter():

    def __init__(self, dataset_dir: str, fold_num = 10, subband=False):
        np.random.seed(42)
        with h5py.File(dataset_dir, "r") as root:
            EEG_group = root["OriginalEEG"] # [subject, channel, time]
            self.total_data = EEG_group["data"][:][..., :84*250]
            if subband:
                print("Sub-frequency Bands Filtering ... ")
                self.total_data = filtrate_all_EEGs(self.total_data)
            self.total_labels = EEG_group["labels"][:]
            self.total_ref_spectrum = root["data"][:]
        
        shuffle_list = np.arange(len(self.total_data))
        np.random.shuffle(shuffle_list)
        self.total_data = self.total_data[shuffle_list]
        self.total_labels = self.total_labels[shuffle_list]
        self.total_ref_spectrum = self.total_ref_spectrum[shuffle_list]

        self.fold = 0
        self.fold_num = fold_num
        self.fold_scale = len(self.total_data) // self.fold_num

    def __iter__(self):
        self.fold = 0
        return self

    def clip(self, start_idx, end_idx):
        train_set = {}
        validate_set = {}
        validate_indices = np.arange(start_idx, end_idx)

        validate_data = self.total_data[validate_indices]
        validate_labels = self.total_labels[validate_indices]
        validate_ref_spectrum = self.total_ref_spectrum[validate_indices]
        validate_set["data"], validate_set["labels"], validate_set["good_list"] = arrange(
                validate_data, validate_labels, validate_ref_spectrum, False)

        train_data = np.delete(self.total_data, validate_indices, axis=0)
        train_labels = np.delete(self.total_labels, validate_indices, axis=0)
        train_ref_spectrum = np.delete(self.total_ref_spectrum, validate_indices, axis=0)
        train_set["data"], train_set["labels"] = arrange(
                train_data, train_labels, train_ref_spectrum, True)

        train_set["data"] = train_set["data"].reshape(train_set["data"].shape[0], -1, train_set["data"].shape[-1])
        validate_set["data"] = validate_set["data"].reshape(validate_set["data"].shape[0], -1, validate_set["data"].shape[-1])
        print(f'In RawEEG: shape={train_set["data"].shape}')
        return train_set, validate_set

    def __next__(self):
        if self.fold >= self.fold_num:
            raise StopIteration
        else:
            train_set: dict = None
            validate_set: dict = None
            if self.fold == 0:
                train_set, validate_set = self.clip(0, self.fold_scale)
            elif self.fold == self.fold_num-1:
                train_set, validate_set = self.clip(self.fold*self.fold_scale, len(self.total_data))
            else:
                train_set, validate_set = self.clip(self.fold*self.fold_scale, (self.fold+1)*self.fold_scale)
            self.fold += 1
            return train_set, validate_set

if __name__ == "__main__":

    base_dir = "/home/xiangnan/E/EDoc/Research/情绪与抑郁识别/Dataset/Depression/Original3Channel"

    EEG_dir = os.path.join(base_dir, "EEG.h5")
    spectrum_dir = os.path.join(base_dir, "spectrum.h5")
    CDFES_dir = os.path.join(base_dir, "global_CDFES.h5")

    shapes = extract_CDAED(EEG_dir, spectrum_dir)
    print(shapes)

# FOLLOWING: FOR INSTANTANEOUS CDFES ANALYSIS
#    CDFES, labels = global_CDFES_computing(spectrum_dir)
#    print(CDFES.shape, labels.shape)
#    with h5py.File(CDFES_dir, "w") as root:
#        root.create_dataset("data", data=CDFES)
#        root.create_dataset("labels", data=labels)
#    print("Global CDFES Dumped.")



