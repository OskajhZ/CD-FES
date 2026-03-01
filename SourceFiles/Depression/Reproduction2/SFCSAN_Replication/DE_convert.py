import os
import math
import numpy as np
from scipy.signal import butter, lfilter
import h5py
from tqdm import tqdm

from typing import *

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

def decompose_a_trial(trial):
    '''
    trial: [3, len]
    '''
    frequency = 250
    bands = ["theta", "alpha", "beta", "gamma"]
    frequency_range = [(4,8), (8, 14), (14, 31), (31, 45)]
    channel_amount, length = trial.shape
    trial_DEs_list = []
    for channel in trial:
        channel_DEs = []
        for band_idx,band in enumerate(bands):
            freq_low, freq_high = frequency_range[band_idx]
            bandpass_channel = butter_bandpass_filter(channel, freq_low, freq_high, frequency, order=3)
            DEs = np.array([])
            for clip_idx in range(length//(3*frequency)):
                clip = bandpass_channel[clip_idx*3*frequency: (clip_idx+1)*3*frequency]
                single_DE = compute_DE(clip)
                DEs = np.append(DEs, single_DE)
            DEs = DEs.reshape(-1, 1) # [clip_amount, 1]
            channel_DEs.append(DEs)
        channel_DEs_arr = np.concatenate(channel_DEs, axis=-1) # [clip_amount, band_amount=4]
        trial_DEs_list.append(channel_DEs_arr[..., np.newaxis])
    trial_DEs = np.concatenate(trial_DEs_list, axis=-1) #[clip_amout, band_amount=4, channel_amount=3]
    return trial_DEs

def compute_feature(EEG_dir: str) -> Tuple[np.ndarray]:
    with h5py.File(EEG_dir, "r") as root:
        EEG_group = root["OriginalEEG"]
        EEGs = EEG_group["data"][:] # [subjects, channels, length]
        labels = EEG_group["labels"][:] # [subjects, ]
    subject_amount = len(labels)

    print("Start DE feature computing:")
    DEs_list = []
    for i in tqdm(range(subject_amount), total=subject_amount):
        subject = EEGs[i]
        subject_DEs = decompose_a_trial(subject)
        DEs_list.append(subject_DEs[np.newaxis, ...])
    DEs = np.concatenate(DEs_list) # [subjects, clip_amount, band_amount=4, channel_amount=3]
    clip_labels = labels[..., np.newaxis]
    clip_amount = DEs.shape[1]
    clip_labels = np.tile(clip_labels, (1, clip_amount)) # [subjects, clip_amount]

    return DEs, clip_labels

if __name__ == '__main__':
    base_folder = "/home/xiangnan/桌面/EDoc/Research/情绪与抑郁识别/Dataset/Depression/Original3Channel"
    EEG_dir = "spectrum.h5"
    DE_save_dir = "DE.h5"

    DEs, labels = compute_feature(os.path.join(base_folder,EEG_dir))
    with h5py.File(os.path.join(base_folder, DE_save_dir), "w") as root:
        root.create_dataset("data", data=DEs)
        root.create_dataset("labels", data=labels)
    print("Shape of data: {}".format(DEs.shape))
    print("Shape of labels: {}".format(labels.shape))

