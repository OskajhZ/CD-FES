'''
Author:
    Xiangnan Zhang: zhangxn@bit.edu.cn
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025
Provides:
    Feature comparison, despite of differential entropy, CD-AED and CD-FES
    Reproductions of Shen 2017 and Cai 2018

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm
import nolds
import multiprocessing

import model_define

np.random.seed(36)

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


def make_statistics(y_pred, y_true) -> dict:
    statis_dict = dict()
    statis_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    statis_dict["precision"] = metrics.precision_score(y_true, y_pred, pos_label=1)
    statis_dict["recall"] = metrics.recall_score(y_true, y_pred, pos_label=1)
    statis_dict["f1"] = metrics.f1_score(y_true, y_pred, pos_label=1)
    return statis_dict


def fir_filter(eeg_signal, cutoff_freqs, fs, window_len=256):
    """
    Blackman-window FIR filter.
    """
    nyq = 0.5 * fs
    low, high = cutoff_freqs
    low_norm, high_norm = low / nyq, high / nyq

    b = signal.firwin(window_len, [low_norm, high_norm], pass_zero=False, window='blackman')
    filtered_signal = signal.filtfilt(b, 1, eeg_signal)
    return filtered_signal


def extract_Cai_features(EEGs) -> np.ndarray:
    '''
    EEGs: [channels, length] for a single subject
    channels = Fp1, Fpz, Fp2
    '''
    fs = 250
    channels, length = EEGs.shape
    l = length // fs
    bands = {
            "theta": (4,8),
            "beta": (14, 30),
            "gamma": (30, 50)
            }

    feature_vector = np.zeros(4)
    fixed_filter = lambda eeg_signal, cutoff_freqs: fir_filter(eeg_signal, cutoff_freqs, fs)
    gamma_of_Fp1 = fixed_filter(EEGs[0], bands["gamma"])
    feature_vector[0] = (gamma_of_Fp1**2).mean()
    theta_of_Fp2 = fixed_filter(EEGs[2], bands["theta"])
    feature_vector[1] = (theta_of_Fp2**2).mean()
    beta_of_Fp2 = fixed_filter(EEGs[2], bands["beta"])
    feature_vector[2] = (beta_of_Fp2**2).mean()
    spectrum = np.fft.fft(EEGs[2], n=length)
    low, high = bands["beta"]
    spectrum_of_beta = np.abs(spectrum[low*l: high*l])
    spectrum_weight = spectrum_of_beta / spectrum_of_beta.sum()
    for i in range(len(spectrum_of_beta)):
        freq = low + i/l
        spectrum_weight[i] *= freq
    feature_vector[3] = spectrum_weight.sum()
    return feature_vector


def Cai_algorithm_preview(EEG_dir) -> dict:
    with h5py.File(EEG_dir, "r") as root:
        EEG_group = root["OriginalEEG"]
        EEGs = EEG_group["data"][:]
        labels = EEG_group["labels"][:]
    Cai_features = np.zeros((len(EEGs), 4))
    print("Extracting Cai Features ... ", end="", flush=True)
    for i, signal in enumerate(EEGs):
        Cai_features[i] = extract_Cai_features(signal)
    print("Done")

    shuffle_list = np.arange(len(EEGs))
    np.random.shuffle(shuffle_list)
    Cai_features = Cai_features[shuffle_list]
    labels = labels[shuffle_list]

    folds = 10
    fold_size = len(Cai_features) // 10
    global_score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    k_series = []
    k = 3
    k_increase = 1
    while k < 40:
        print("k={}:".format(k))
        score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        print("Applying ten folds validation ... ", end="", flush=True)
        for i in range(folds):
            start = fold_size*i
            end = fold_size*(i+1)
            if end > len(Cai_features):
                end = len(Cai_features)
            validate_list = shuffle_list[start: end]
            validate_data = Cai_features[validate_list]
            validate_labels = labels[validate_list]
            train_data = np.delete(Cai_features, validate_list, axis=0)
            train_labels = np.delete(labels, validate_list, axis=0)
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            validate_data = scaler.transform(validate_data)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(train_data, train_labels)
            y_pred = knn.predict(validate_data)
            fold_result = make_statistics(y_pred, validate_labels)
            for key in fold_result.keys():
                score_dict[key].append(fold_result[key])
        print("Done")
        k_series.append(k)
        if k == 5:
            print(score_dict)
        k += k_increase
        for key in score_dict.keys():
            arrtype_scores = np.array(score_dict[key])
            avg_score = arrtype_scores.mean()
            global_score_dict[key].append(avg_score)
    fig = plt.figure(figsize=(20, 12))
    for i,key in enumerate(global_score_dict.keys()):
        plt.subplot(2,2,i+1)
        plt.plot(k_series, global_score_dict[key])
        plt.title("{} with each k".format(key), fontsize=20)
        plt.xlabel("k", fontsize=15)
        plt.ylabel("%", fontsize=15)
    plt.tight_layout()
    fig.savefig("Cai_replication")
    plt.show()

    return global_score_dict


def Cai_algorithm(EEG_dir, classifier="KNN", classifier_kwargs={}) -> dict:
    with h5py.File(EEG_dir, "r") as root:
        EEG_group = root["OriginalEEG"]
        EEGs = EEG_group["data"][:]
        labels = EEG_group["labels"][:]
    Cai_features = np.zeros((len(EEGs), 4))
    print("Extracting Cai Features ... ", end="", flush=True)
    for i, signal in enumerate(EEGs):
        Cai_features[i] = extract_Cai_features(signal)
    print("Done")

    shuffle_list = np.arange(len(EEGs))
    np.random.shuffle(shuffle_list)
    Cai_features = Cai_features[shuffle_list]
    labels = labels[shuffle_list]

    folds = 10
    fold_size = len(Cai_features) // folds
    score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for i in range(folds):
        start = fold_size*i
        end = fold_size*(i+1)
        if end > len(Cai_features):
            end = len(Cai_features)
        validate_list = shuffle_list[start: end]
        validate_data = Cai_features[validate_list]
        validate_labels = labels[validate_list].reshape(-1)
        train_data = np.delete(Cai_features, validate_list, axis=0)
        train_labels = np.delete(labels, validate_list, axis=0).reshape(-1)

        _, fold_result = model_define.apply_classifier(
                (train_data, train_labels),
                (validate_data, validate_labels),
                1,
                classifier,
                **classifier_kwargs
                )

        for key in fold_result.keys():
            score_dict[key].append(fold_result[key])
    return score_dict


def gaussian_kernel(mean: np.ndarray, variance: float) -> np.ndarray:
    '''
    G(x - x_i, sigma^2) = 1/sqrt(2*pi*sigma^2) * exp(-||x-x_i||^2 / (2*sigma^2))
    '''
    k = 1/math.sqrt(2*math.pi*variance)
    index = -mean**2/(2*variance)
    exponent = np.exp(index)
    return k*exponent


def Renyi_entropy(x: np.ndarray) -> float:
    std = x.std()
    sub_x_list = []
    for i, point in enumerate(x):
        if i == len(x)-1:
            break
        rest = x[i+1:]
        residual = -(rest-point)
        sub_x_list.append(residual)
    residuals = np.concatenate(sub_x_list)
    gaussian_arr = gaussian_kernel(residuals, 2*std**2)
    sum_of_gaussian = 2 * gaussian_arr.sum()
    exponent = 1/len(x)**2 * sum_of_gaussian
    entropy = -math.log(exponent, 2)
    return entropy


def C0_complexity(x) -> float:
    spectrum = np.fft.fft(x)
    avg_energy = (abs(spectrum)**2).sum()/len(spectrum)
    regular_spectrum = spectrum.copy()
    below_list = (abs(spectrum)**2 <= avg_energy)
    regular_spectrum[below_list] = 0
    regular = np.fft.ifft(regular_spectrum)
    irregular = x - regular
    C0 = (np.abs(irregular)**2).sum() / (np.abs(x)**2).sum()
    return C0


def full_FIR_filter(EEGs, bands=None):
    '''
    EEGs: [channels, length]
    '''
    fs = 250
    channels, length = EEGs.shape
    if bands is None:
        bands = {
                "delta": (1, 4),
                "theta": (4,8),
                "alpha": (8, 13),
                "beta": (14, 30)
                }

    fixed_filter = lambda eeg_signal, cutoff_freqs: fir_filter(eeg_signal, cutoff_freqs, fs)
    channel_list = []
    for channel in EEGs:
        band_list = []
        band_list.append(channel[np.newaxis, ...])
        for key in bands.keys():
            singleband = fixed_filter(channel, bands[key])
            band_list.append(singleband[np.newaxis, ...])
        filtered_channel = np.concatenate(band_list)
        channel_list.append(filtered_channel[np.newaxis, ...])
    filtered = np.concatenate(channel_list)
    return filtered


def extract_Shen2017_features(EEGs) -> np.ndarray:
    '''
    EEGs: [channels, bands, length] for a single subject
    channels = Fp1, Fpz, Fp2
    '''
    feature_list = []
    for channel in EEGs:
        for band in channel:
            spectrum = np.abs(np.fft.fft(band))
            max_frequency = np.argmax(spectrum)
            feature_list.append(max_frequency)
            freqs = np.arange(len(spectrum))
            weights = spectrum / spectrum.sum()
            centroid_frequency = (freqs*weights).sum()
            feature_list.append(centroid_frequency)
            r_entropy = Renyi_entropy(band)
            feature_list.append(r_entropy)
            corr_dim  = nolds.corr_dim(band, emb_dim=5)
            feature_list.append(corr_dim)
            C0 = C0_complexity(band)
            feature_list.append(C0)

    feature_vector = np.array(feature_list)
    return feature_vector


def expand_shen_features(signal):
    feature = extract_Shen2017_features(signal)
    feature = feature[np.newaxis, :]
    return feature


def chunk(EEGs, labels, clip_length):
    '''
    EEGs: [subjects, channels, ..., length]
    '''
    org_length = EEGs.shape[-1]
    subjects = EEGs.shape[0]
    clip_amount = org_length // clip_length
    EEGs = EEGs[..., :clip_amount*clip_length]
    EEGs = EEGs.swapaxes(1, -1)
    retain_shape = EEGs.shape[2:]
    EEGs = EEGs.reshape(subjects, clip_amount, clip_length, *retain_shape)
    EEGs = EEGs.swapaxes(2, -1)
    labels = labels[:, np.newaxis]
    labels = np.repeat(labels, clip_amount, axis=1)
    return EEGs, labels


def Shen2017_algorithm(EEG_dir, classifier="SVM", compute_features=True, classifier_kwargs={}) -> dict:
    if compute_features:
        with h5py.File(EEG_dir, "r") as root:
            EEG_group = root["OriginalEEG"]
            EEGs = EEG_group["data"][:]
            labels = EEG_group["labels"][:]
        filtered_list = []
        for subject in EEGs:
            filtered_subject = full_FIR_filter(subject)
            filtered_list.append(filtered_subject[np.newaxis, ...])
        EEGs = np.concatenate(filtered_list)
        EEGs, labels = chunk(EEGs, labels, 250*5)
        print("Extracting Shen Features ... ", end="", flush=True)
        process_pool = multiprocessing.Pool(1)
        subject_list = []
        for subject in tqdm(EEGs):
            subject_features = process_pool.map(expand_shen_features, subject)
            subject_features = np.concatenate(subject_features)[np.newaxis, :]
            subject_list.append(subject_features)
        Shen_features = np.concatenate(subject_list)
        print("Done")
        with h5py.File("Shen2017_feature.hdf5", "w") as root:
            root.create_dataset("data", data=Shen_features)
            root.create_dataset("labels", data=labels)
        print("Feature saved")

    with h5py.File("Shen2017_feature.hdf5", "r") as root:
        Shen_features = root["data"][:]
        labels = root["labels"][:]

    shuffle_list = np.arange(len(labels))
    np.random.shuffle(shuffle_list)
    Shen_features = np.real(Shen_features[shuffle_list])
    labels = labels[shuffle_list]

    folds = 10
    fold_size = len(Shen_features) // folds
    score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    clips_per_subject = Shen_features.shape[1]

    for i in range(folds):
        start = fold_size*i
        end = fold_size*(i+1)
        if end > len(Shen_features):
            end = len(Shen_features)
        validate_list = shuffle_list[start: end]
        n_valid_subjects = len(validate_list)
        clip_shape = Shen_features.shape[2:]
        validate_data = Shen_features[validate_list].reshape(-1, *clip_shape)
        validate_labels = labels[validate_list].reshape(-1)
        train_data = np.delete(Shen_features, validate_list, axis=0).reshape(-1, *clip_shape)
        train_labels = np.delete(labels, validate_list, axis=0).reshape(-1)

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        validate_data = scaler.transform(validate_data)

        if classifier == "SVM":
            clf = SVC(**classifier_kwargs)
        elif classifier == "KNN":
            clf = KNeighborsClassifier(**classifier_kwargs)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        clf.fit(train_data, train_labels)
        pred = clf.predict(validate_data)

        ensemble_pred = []
        ensemble_labels = []
        for s in range(n_valid_subjects):
            start_clip = s * clips_per_subject
            end_clip = (s + 1) * clips_per_subject
            subject_pred = pred[start_clip:end_clip]
            subject_label = labels[validate_list[s], 0]
            subject_ensemble = 1 if subject_pred.mean() >= 0.5 else 0
            ensemble_pred.append(int(subject_ensemble))
            ensemble_labels.append(int(subject_label))

        fold_result = make_statistics(np.array(ensemble_pred), np.array(ensemble_labels))
        for key in fold_result.keys():
            score_dict[key].append(fold_result[key])

    return score_dict


def extract_full_features(EEGs) -> np.ndarray:
    '''
    EEGs: [channels, bands, length] for a single subject
    channels = Fp1, Fpz, Fp2
    '''
    feature_names = ["max_frequency", "centroid_frequenct", "absolute_power",
            "Renyi_entropy", "correlation_dimension", "C0_complexity"]
    feature_names = np.array(feature_names)
    channels, bands, length = EEGs.shape
    features = np.zeros([channels, bands, len(feature_names)])
    for channel_idx,channel in enumerate(EEGs):
        for band_idx,band in enumerate(channel):
            spectrum = np.abs(np.fft.fft(band))**2

            max_frequency = np.argmax(spectrum)
            features[channel_idx, band_idx, 0] = max_frequency

            freqs = np.arange(len(spectrum))
            weights = spectrum / spectrum.sum()
            centroid_frequency = (freqs*weights).sum()
            features[channel_idx, band_idx, 1] = centroid_frequency

            abs_power = spectrum.sum()
            features[channel_idx, band_idx, 2] = abs_power

            r_entropy = Renyi_entropy(band)
            features[channel_idx, band_idx, 3] = r_entropy

            corr_dim  = nolds.corr_dim(band, emb_dim=5)
            features[channel_idx, band_idx, 4] = corr_dim

            C0 = C0_complexity(band)
            features[channel_idx, band_idx, 5] = C0

    features = features[np.newaxis]
    return features, feature_names


def full_features_extract_and_dump(EEG_dir, save_dir="full_feature.hdf5"):
    with h5py.File(EEG_dir, "r") as root:
        EEG_group = root["OriginalEEG"]
        EEGs = EEG_group["data"][:]
        labels = EEG_group["labels"][:]
    filtered_list = []
    bands = {
            "theta": (4,8),
            "alpha": (8, 14),
            "beta": (14, 31),
            "gamma": (31, 40)
            }
    for subject in EEGs:
        filtered_subject = full_FIR_filter(subject, bands)[:, 1:]
        filtered_list.append(filtered_subject[np.newaxis, ...])
    EEGs = np.concatenate(filtered_list)
    EEGs, labels = chunk(EEGs, labels, 250*3)
    print("Extracting Features ... ", end="", flush=True)
    process_pool = multiprocessing.Pool(1)
    subject_list = []
    names = None

    for i, subject in tqdm(enumerate(EEGs), total=len(EEGs)):
        subject_features_names = process_pool.map(extract_full_features, subject)
        subject_features = [feature for feature,_ in subject_features_names]
        if i==0:
            _, names = subject_features_names[0]
        subject_features = np.concatenate(subject_features)[np.newaxis, :]
        subject_list.append(subject_features)
    full_features = np.concatenate(subject_list)
    print("Done")

    with h5py.File(save_dir, "w") as root:
        dataset = root.create_dataset("data", data=full_features)
        dataset.attrs["feature_names"] = str(names)
        root.create_dataset("labels", data=labels)
    print("Feature saved as \"{}\".".format(save_dir))


def metrices_avg(metrices):
    for key, value in metrices.items():
        value_arr = np.array(value)
        avg = value_arr.mean()
        std = value_arr.std(ddof=1)
        se = std / math.sqrt(10)
        avg = round(avg*100, 2)
        se = round(se*100, 2)
        print(r"For {}: {}$\pm${} %".format(key, avg, se))
    return None


def feature_experiment_one_by_one(feature_dir):
    feature_names = ["max_frequency", "centroid_frequenct", "absolute_power",
            "Renyi_entropy", "correlation_dimension", "C0_complexity"]
    classifiers = ["SVM", "KNN", "DT", "RF", "XGBoost"]

    with h5py.File(feature_dir, "r") as root:
        features = root["data"][:]
        labels = root["labels"][:]

    shuffle_list = np.arange(len(labels))
    np.random.shuffle(shuffle_list)
    features = features[shuffle_list]
    labels = labels[shuffle_list]

    for i, name in enumerate(feature_names):
        sub_features = features[..., i]
        shape = sub_features.shape
        sub_features = sub_features.reshape(shape[0], shape[1], -1)
        print("{}:".format(name))

        folds = 10
        fold_size = len(sub_features) // folds

        for i, classifier in enumerate(classifiers):
            print("    On {}".format(classifier))
            score_dict = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            for i in range(folds):
                start = fold_size*i
                end = fold_size*(i+1)
                if end > len(sub_features):
                    end = len(sub_features)
                validate_list = shuffle_list[start: end]
                clip_shape = sub_features.shape[2:]
                validate_data = sub_features[validate_list].reshape(-1, *clip_shape)
                validate_labels = labels[validate_list].reshape(-1)
                train_data = np.delete(sub_features, validate_list, axis=0).reshape(-1, *clip_shape)
                train_labels = np.delete(labels, validate_list, axis=0).reshape(-1)

                _, fold_result = model_define.apply_classifier(
                        (train_data, train_labels),
                        (validate_data, validate_labels),
                        100,
                        classifier
                        )

                for key in fold_result.keys():
                    score_dict[key].append(fold_result[key])

            print("        Raw result: {}".format(score_dict))
            for key, value in score_dict.items():
                value_arr = np.array(value)
                avg = value_arr.mean()
                std = value_arr.std(ddof=1)
                se = std / math.sqrt(10)
                avg = round(avg*100, 2)
                se = round(se*100, 2)
                print(r"        For {}: {}$\pm${} %".format(key, avg, se))

    return None


if __name__ == "__main__":
    import average

    EEG_dir = "/home/xiangnan/E/EDoc/Research/Dataset/MODMA/EEG_128channels_resting_lanzhou_2015/preprocessed/spectrum.h5"
    classifiers = ["SVM", "KNN"]
    svm_kwargs = {'kernel': 'sigmoid', 'C': 0.1, 'gamma': 0.1, 'class_weight': 'balanced', 'random_state': 42}
    knn_kwargs = {'n_neighbors': 3, 'weights': 'uniform', 'p': 2}

    for i,classifier in enumerate(classifiers):
        compute_features = False
        if classifier == "SVM":
            result = Shen2017_algorithm(EEG_dir, classifier, compute_features, classifier_kwargs=svm_kwargs)
        elif classifier == "KNN":
            result = Shen2017_algorithm(EEG_dir, classifier, compute_features, classifier_kwargs=knn_kwargs)
        print("Shen2017 on {} raw: {}".format(classifier, result))
        avg_dict, se_dict = average.compute_avg_se(result, False)
        average.latex_output(avg_dict, se_dict)

    feature_experiment_one_by_one("full_feature.hdf5")
