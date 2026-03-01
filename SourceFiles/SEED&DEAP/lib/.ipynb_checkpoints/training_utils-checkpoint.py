'''
By Xiangnan Zhang, 2025
    School of Future Technologies, Beijing Institute of Technology
'''

import shutil
import math
import warnings
from typing import Tuple, Callable, List

from tqdm import tqdm
import numpy as np
from sklearn import metrics
import h5py

import torch
from torch.utils.tensorboard import SummaryWriter

from . import _base

torch.manual_seed(42)
np.random.seed(42)

# Following: Metrics

def make_statistics(y_pred, y_true) -> dict:
    statis_dict = dict()
    statis_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    statis_dict["precision"] = metrics.precision_score(y_true, y_pred, pos_label=1, average="weighted")
    statis_dict["recall"] = metrics.recall_score(y_true, y_pred, pos_label=1, average="weighted")
    statis_dict["f1"] = metrics.f1_score(y_true, y_pred, pos_label=1, average="weighted")
    return statis_dict

def output_avg_se(statis):
    '''
    According to the statis (metrics),
    output the average metrics with standard errors.
    '''

    converted = {"accuracy": [],
                 "precision": [],
                 "recall": [],
                 "f1": []}
    for fold in statis:
        for key in converted.keys():
            converted[key].append(fold[key])

    statis = converted

    avg_dict = dict()
    se_dict = dict()

    for key, value in statis.items():
        value_arr = np.array(value)
        avg = value_arr.mean()
        std = value_arr.std(ddof=1)
        se = std / math.sqrt(10)
        avg = round(avg*100, 2)
        se = round(se*100, 2)
        avg_dict[key] = avg
        se_dict[key] = se

    for key in avg_dict.keys():
        print(r"For {}: {}$\pm${} %".format(key, avg_dict[key], se_dict[key]))

    return avg_dict, se_dict


def make_ensemble(pred, labels, ensemble_amount: int, good_list=None, judgement: str="category") -> Tuple[np.ndarray]:
    '''
    judgement: "category" or "probability" or "logits"
        if "category": consider positive only if 50% predicted category is positive.
        if "probability": consider positive if the average predicted probability is above 50%
    '''
    def category_based_judgement(pred_clip):
        pred_num = pred_clip.sum() / len(pred_clip)
        if pred_num >= 0.5:
            pred_num=1
        else:
            pred_num=0
        return pred_num
    def probability_based_judgement(pred_clip):
        avg_probability = pred_clip.mean()
        if avg_probability >= 0.5:
            return 1
        else:
            return 0
    def logits_based_judgement(pred_clip):
        mid_diff = pred_clip[..., 0] - pred_clip[..., 1]
        mid_diff = mid_diff.mean(0)
        if mid_diff < 0:
            return 1
        else:
            return 0

    if good_list is None:
        good_list = np.ones(len(labels)).astype(bool)

    ensemble_pred = []
    ensemble_labels = []
    idx = 0
    while idx+ensemble_amount <= len(pred):
        pred_clip = pred[idx: idx+ensemble_amount]
        good_indices = good_list[idx: idx+ensemble_amount]
        pred_clip = pred_clip[good_indices]
        labels_clip = labels[idx: idx+ensemble_amount]
        pred_num: int = None
        if judgement == "category":
            pred_num = category_based_judgement(pred_clip)
        elif judgement == "probability":
            pred_num = probability_based_judgement(pred_clip)
        elif judgement == "logits":
            pred_num = logits_based_judgement(pred_clip)
        else:
            raise ValueError("parameter \"judgenment\" should be \"category\" or \"probability\".")
        ensemble_pred.append(pred_num)
        labels_num = labels_clip.sum() / ensemble_amount
        if labels_num!=0 and labels_num!=1:
            raise ValueError("Esemble amount error. The {}th labels clip is {}".format(idx, list(labels_clip)))
        else:
            ensemble_labels.append(int(labels_num))
        idx += ensemble_amount
    ensemble_pred = np.array(ensemble_pred)
    ensemble_labels = np.array(ensemble_labels)
    return ensemble_pred, ensemble_labels

class StaticSet(torch.utils.data.Dataset):

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, apply_augment=False):
        self.data = data
        self.targets = targets
        self.apply_augment = apply_augment
    def set_augment(self, mode: bool):
        self.apply_augment = mode
    def __getitem__(self,idx):
        single_data = self.data[idx]
#        if self.apply_augment:
#            single_data = augment(single_data)
        return single_data, self.targets[idx]
    def __len__(self):
        return len(self.targets)

# Following: Trainer define

class NetworkTrainer():

    def __init__(self, model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            test_loader, test_good_list = None,
            validate_loader = None, validate_good_list = None,
            ensemble_amount = 1,
            ensemble_judgement = "probability",
            lr = 1e-4,
            trainer_idx = 0):
        '''
        Ensemble mode will be used only when ensemble_amount > 1.
        If validate_loader is None: experiment with fixed epoch will be done,
            or else select the model with the highest validation F1 score.
        '''
        self.model = model
        if torch.cuda.is_available():
            print("Use CUDA")
            self.model.to("cuda")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validate_loader = validate_loader
        self.ensemble_amount = ensemble_amount
        self.ensemble_judgement = ensemble_judgement
        self.test_good_list = test_good_list
        self.validate_good_list = validate_good_list
        self.trainer_idx = trainer_idx

        self.dir = "logs/log_{}".format(trainer_idx)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.log = SummaryWriter(log_dir = self.dir)
        print("log location: {}".format(self.dir))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        print("Optimizer configuration:")
        print(self.optimizer)

        print("Trainer Initialization Finished.")

    def get_category(self, logits):
        return torch.argmax(logits, dim=1)

    def get_probability(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs

    def save(self):
        torch.save(self.model.state_dict(),
                "model_state_dict_{}.pt".format(self.trainer_idx))

    def load(self):
        state_dict = torch.load("model_state_dict_{}.pt".format(self.trainer_idx))
        self.model.load_state_dict(state_dict)

    def eval(self, data_type = "test", eval_model: torch.nn.Module = None):
        '''
        data_type: "test" to use the test set, "validation" to use the validation set.
        eval_model: model used for evaluation. Default current model (self.model)
        '''
        if eval_model is None:
            eval_model = self.model
        eval_model.eval()
        avg_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []

        data_loader, good_list = None, None
        if data_type == "test":
            data_loader = self.test_loader
            good_list = self.test_good_list
        elif data_type == "validation":
            data_loader = self.validate_loader
            good_list = self.validate_good_list
        else:
            raise ValueError("Invalid data_type, got: {}".format(data_type))

        with torch.no_grad():
            for x,y in data_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                tpred_logits = eval_model(x, True)
                loss = self.loss_fn(tpred_logits, y)
                avg_loss += loss.cpu().item()*len(y)
                pred = self.get_category(tpred_logits)
                all_preds.append(pred)
                all_labels.append(y)
                all_logits.append(tpred_logits)

        all_preds = torch.cat(all_preds).cpu()
        all_labels = torch.cat(all_labels).cpu()
        avg_loss /= len(self.validate_loader.dataset)
        segment_result = make_statistics(all_preds.cpu().numpy(), all_labels.cpu().numpy())
        segment_result["loss"] = avg_loss

        ensemble_result, confusion_matrix = None, None
        if self.ensemble_amount > 1:
            all_logits = torch.cat(all_logits)
            ensemble_preds, ensemble_labels = make_ensemble(
                    all_logits, all_labels, self.ensemble_amount,
                    good_list, judgement = self.ensemble_judgement)
            ensemble_result = make_statistics(ensemble_preds, ensemble_labels)
            if data_type == "test":
                confusion_matrix = metrics.confusion_matrix(ensemble_labels, ensemble_preds)
        elif data_type == "test":
            confusion_matrix = metrics.confusion_matrix(all_preds, all_labels)

        return segment_result, ensemble_result, confusion_matrix

    def train_one_epoch(self):
        '''
        train for one epoch
        '''
        self.model.train()
        loss, pred_logits, y = None, None, None
        for x,y in self.train_loader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred_logits = self.model(x)
            loss = self.loss_fn(pred_logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss = loss.detach().cpu().item()
        pred = self.get_category(pred_logits)
        accuracy = metrics.accuracy_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
        return loss, accuracy

    def train_with_fixed_epochs(self, epochs = 10) -> None:
        train_loss, train_acc = None, None
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss, train_acc = self.train_one_epoch()
            self.log.add_scalar("train/loss", train_loss, epoch+1)
            self.log.add_scalar("train/accuracy", train_acc, epoch+1)

            segment_result, ensemble_result, _ = self.eval("test")
            for key,value in segment_result.items():
                self.log.add_scalar(f"test_segment/{key}", value, epoch+1)
            if self.ensemble_amount > 1:
                for key,value in ensemble_result.items():
                    self.log.add_scalar(f"test_ensemble/{key}", value, epoch+1)

        print("Training finished.")
        print("Training loss: {}, training accuracy: {}".format(
            train_loss, train_acc)
            )
        self.save()

    def train_with_validation(self, epochs = 10) -> None:
        highest_val_f1 = 0
        train_loss, train_acc = None, None
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss, train_acc = self.train_one_epoch()
            self.log.add_scalar("train/loss", train_loss, epoch+1)
            self.log.add_scalar("train/accuracy", train_acc, epoch+1)

            segment_result, ensemble_result, _ = self.eval("validation")
            validate_result = segment_result
            for key,value in segment_result.items():
                self.log.add_scalar(f"validate_segment/{key}", value, epoch+1)
            if self.ensemble_amount > 1:
                for key,value in ensemble_result.items():
                    self.log.add_scalar(f"validate_ensemble/{key}", value, epoch+1)
                validate_result = ensemble_result
            if validate_result["f1"] >= highest_val_f1:
                highest_val_f1 = validate_result["f1"]
                self.save()
        print("Training finished.")
        print("Training loss: {}, training accuracy: {}".format(
            train_loss, train_acc)
            )
        print("Highest validation F1: {}".format(highest_val_f1))
        self.load()

    def execute(self, epochs = 10):
        if self.validate_loader is None:
            self.train_with_fixed_epochs(epochs)
        else:
            self.train_with_validation(epochs)
        segment_result, ensemble_result, confusion_matrix = self.eval(
                data_type = "test")
        return segment_result, ensemble_result, confusion_matrix
    
def permute_timedim_first(signals):
    '''
    signals: [num, ..., timestep]
    return: [num, timestep, ...]
    '''
    dim_list = list(range(len(signals.shape)))
    dim_tuple = tuple(dim_list)
    reshape_dim = (dim_tuple[0], dim_tuple[-1], *dim_tuple[1:-1])
    permuted_signals = np.transpose(signals, axes=reshape_dim)
    return permuted_signals

def standardize(train_x, validate_x, mean=0, std=1, time_avg=True):
    '''
    x: [num, ..., length]
    time_avg: Average through time steps first. Suit the in-oscillation signals.
        Or else treating each time step independently.
    '''
    reduced_data = None
    if time_avg:
        reduced_data = train_x.mean(-1)
    else:
        reduced_data = permute_timedim_first(train_x)
        channel_shape = reduced_data.shape[2:]
        reduced_data = reduced_data.reshape(-1, *channel_shape)
    means = reduced_data.mean(0)[..., np.newaxis]
    stds = reduced_data.std(0)[..., np.newaxis]
    length = train_x.shape[-1]
    means = np.tile(means, (1, 1, length))
    stds = np.tile(stds, (1, 1, length))
    train_x = (train_x-means)/stds
    validate_x = (validate_x-means)/stds

    train_x, validate_x = train_x*std, validate_x*std
    train_x, validate_x = (train_x+mean), (validate_x+mean)

    return train_x, validate_x

class SequenceStandardizer():

    def __init__(self, obj_mean = 0, obj_std = 1, time_avg = False):
        self.obj_mean = obj_mean
        self.obj_std = obj_std
        self.time_avg = time_avg
        self.coef_mean = None
        self.coef_std = None
    
    def fit_transform(self, train_x):
        self.fit(train_x)
        transformed = self.transform(train_x)
        return transformed

    def fit(self, train_x) -> None:
        reduced_data = None
        if self.time_avg:
            reduced_data = train_x.mean(-1)
        else:
            reduced_data = permute_timedim_first(train_x)
            channel_shape = reduced_data.shape[2:]
            reduced_data = reduced_data.reshape(-1, *channel_shape)
        means = reduced_data.mean(0)[..., np.newaxis]
        stds = reduced_data.std(0)[..., np.newaxis]
        length = train_x.shape[-1]
        self.coef_mean = np.tile(means, (1, 1, length))
        self.coef_std = np.tile(stds, (1, 1, length))

    def transform(self, x):
        if self.coef_mean is None or self.coef_std is None:
            raise RuntimeError("SequenceStandardizer.fit() or .fit_transform() should be called first.")
        x = (x-self.coef_mean)/self.coef_std
        x = x * self.obj_std + self.obj_mean
        return x

def segment(sequence, label, window=384):
    '''
    sequence: [num, ... timestep]
    label: [num]
    return:
        segmented_sequence: [num*(timestep//window), ...., window]
        label: [num*(timestep//window)]
    '''
    length = sequence.shape[-1]
    clip_num = length // window

    sequence = sequence[..., :clip_num*window]
    sequence = sequence.reshape(*sequence.shape[:-1], clip_num, window)
    axes = tuple(range(len(sequence.shape)))
    sequence = np.transpose(sequence,
            axes=(axes[0], axes[-2], *axes[1:-2], axes[-1])) # [num, clip_num, ..., window]
    sequence = sequence.reshape(
            sequence.shape[0]*sequence.shape[1],
            *sequence.shape[2:]) # Clips for a spicific subject are still continuous

    label = label[:, np.newaxis]
    label = np.repeat(label, clip_num, axis=-1)
    label = label.reshape(-1)
    return sequence, label

def encode_dataset_with_BSA(train_data, test_data, avg_time=False):
    train_data, test_data = standardize(train_data, test_data, 0, 1/3, avg_time)
    train_cshape = train_data.shape[:-1]
    test_cshape = test_data.shape[:-1]
    length = train_data.shape[-1]
    train_data = train_data.reshape(-1, length)
    test_data = test_data.reshape(-1, length)
    encoder = _base.BSAEncoder()
    _, _, fig = encoder.test(test_data[..., :200])
    train_data = encoder.encode(train_data).reshape(*train_cshape, length)
    test_data = encoder.encode(test_data).reshape(*test_cshape, length)
    fig.savefig("BSA_preview.jpg")
    return train_data, test_data

def encode_standardized_sequence_with_BSA(data, visualize=False):
    data_cshape = data.shape[:-1]
    length = data.shape[-1]
    data = data.reshape(-1, length)
    encoder = _base.BSAEncoder()
    if visualize:
        _, _, fig = encoder.test(data)
        fig.savefig("BSA_preview.jpg")
    data = encoder.encode(data).reshape(*data_cshape, length)
    return data

class BSASetEncoder():
    '''
    Combine the sdandardization and encoding together.
    '''

    def __init__(self):
        self.encoder = _base.BSAEncoder()
        self.standardizer = SequenceStandardizer(obj_mean=0, obj_std=1/3, time_avg=False)

    def fit_transform(self, data):
        '''
        Method for the training dataset
        '''
        self.standardizer.fit(data)
        data = self.transform(data, visualize=False)
        return data

    def transform(self, data, visualize=False):
        '''
        Method for evaluation and testing.
        '''
        data = self.standardizer.transform(data)
        data_cshape = data.shape[:-1]
        length = data.shape[-1]
        data = data.reshape(-1, length)
        if visualize:
            SNR, firing_ratio, fig = self.encoder.test(data)
            fig.savefig("BSA_preview.jpg")
            print("BSA test result: SNR = {}, firing ratio = {}".format(SNR, firing_ratio))
        data = self.encoder.encode(data).reshape(*data_cshape, length)
        return data 

def shuffle_evenly(data, labels):
    '''
    To make data belonging to different classes shuffled evenly.
    The tail part of the shuffled data will be even the most,
    thus to make the training set better.
    data: [num, ...]
    label: start by 0, [num,]
    '''
    amount = len(data)
    shuffle_list = np.arange(amount)
    np.random.shuffle(shuffle_list)
    data = data[shuffle_list]
    labels = labels[shuffle_list]

    category_amount = int(np.max(labels).item()) + 1
    categorized_data_list = []
    ptr_list = [0 for _ in range(category_amount)]
    for category in range(category_amount):
        data_in_category = data[labels==category]
        categorized_data_list.append(data_in_category)

    shuffled_data = np.empty_like(data)
    shuffled_labels = np.empty_like(labels)
    fill_ptr = amount - 1

    while fill_ptr >= 0:
        for category in range(category_amount):
            category_ptr = ptr_list[category]
            data_in_category = categorized_data_list[category]
            if category_ptr < len(data_in_category) and fill_ptr >= 0:
                shuffled_data[fill_ptr] = data_in_category[category_ptr]
                shuffled_labels[fill_ptr] = category
                ptr_list[category] += 1
                fill_ptr -= 1

    return shuffled_data, shuffled_labels

class CrossValidationIter():

    def __init__(self,
            task_fEEG: np.ndarray, label: np.ndarray,
            preprocess_function: Callable[[np.ndarray], np.ndarray],
            fold_num = 10,
            mean = 5,
            std = 2.5,
            segment_window = 384,
            encode = True,
            avg_time = True):
        '''
        EEGs should be filtrated_first.
        fEEG represents filtrated EEG
        Shape of EEG: [num, (band, electrode,) timestep]
        avg_time: whether average signals through time when standardizing.
        '''
        self.task_fEEG = task_fEEG
        self.label = label
        self.mean = mean
        self.std = std
        self.segment_window = segment_window
        self.preprocess_function = preprocess_function
        self.encode = encode
        self.avg_time = avg_time

        amount = len(self.task_fEEG)
        self.task_fEEG, self.label = shuffle_evenly(self.task_fEEG, self.label)

        self.fold = 0
        self.fold_num = fold_num
        self.fold_scale = amount // self.fold_num

        self.total_amount = amount

    def __iter__(self):
        self.fold = 0
        return self

    def clip(self, start_idx, end_idx):
        validate_indices = np.arange(start_idx, end_idx)

        test_task = self.task_fEEG[validate_indices]
        test_label = self.label[validate_indices]

        train_task = np.delete(self.task_fEEG, validate_indices, axis=0)
        train_label = np.delete(self.label, validate_indices, axis=0)

        train_data, test_data = self.preprocess_function(train_task, test_task)

        if self.encode:
            train_data, test_data = encode_dataset_with_BSA(train_data, test_data, self.avg_time)
        else:
            train_data, test_data = standardize(train_data, test_data, self.mean, self.std, self.avg_time)

        train_data, train_label = segment(train_data, train_label, self.segment_window)
        test_data, test_label = segment(test_data, test_label, self.segment_window)

        train_set = {"data": train_data, "labels": train_label}
        test_set = {"data": test_data, "labels": test_label}
        return train_set, test_set

    def __next__(self):
        if self.fold >= self.fold_num:
            raise StopIteration
        else:
            train_set: dict = None
            test_set: dict = None
            if self.fold == 0:
                train_set, test_set = self.clip(0, self.fold_scale)
            elif self.fold == self.fold_num-1:
                train_set, test_set = self.clip(
                        self.fold*self.fold_scale,
                        self.total_amount)
            else:
                train_set, test_set = self.clip(
                        self.fold*self.fold_scale,
                        (self.fold+1)*self.fold_scale)
            self.fold += 1
            return train_set, test_set, None # No validation set provided

class ISIter(): # Pending the encoding logic
    '''
    Iterator for independent-subject scenario (or so-called subject dependent).
    Independent-subject: train the classifier for only one specific subject.
    Leave-one-subject-out (LOVO): cross subject.
    The scenario criterion is referred to LibEER (Liu 2025).
    '''

    def __init__(self,
            task_fEEG: np.ndarray, label: np.ndarray,
            preprocess_function: Callable[[np.ndarray], np.ndarray],
            train_test_val_ratio = (0.85, 0.15, 0),
            mean = 5,
            std = 2.5,
            segment_window = 384,
            encode = True,
            avg_through_time = True):
        '''
        EEGs should be filtrated_first.
        fEEG represents filtrated EEG
        Shape of EEG: [subject, trial, band, electrode, timestep]
        Shape of labels: [subject, trial]
        set train_test_val_ratio = (x, x, 0) if validation set is not needed.
        '''
        self.task_fEEG = task_fEEG
        self.label = label
        self.train_test_val_ratio = train_test_val_ratio
        ratio_sum = train_test_val_ratio[0] + train_test_val_ratio[1] + train_test_val_ratio[2]
        if ratio_sum != 1:
            raise ValueError(
                    "The sum of train, test and validation ratio is not equal to 1: got {}".format(ratio_sum))
        self.mean = mean
        self.std = std
        self.segment_window = segment_window
        self.preprocess_function = preprocess_function
        self.encode = encode
        self.avg_through_time = avg_through_time

        self.subject_idx = 0
        self.total_subject = self.task_fEEG.shape[0]

    def __iter__(self):
        self.subject_idx = 0
        return self

    def __next__(self):
        if self.subject_idx >= self.total_subject:
            raise StopIteration
        subject_data = self.task_fEEG[self.subject_idx]
        subject_label = self.label[self.subject_idx]

        subject_data, subject_label = shuffle_evenly(subject_data, subject_label)

        amount = len(subject_data)
        test_amount = int(amount * self.train_test_val_ratio[1])
        val_amount = int(amount * self.train_test_val_ratio[2])
        if val_amount == 0 and self.train_test_val_ratio[2] != 0:
            warnings.warn("Though the validation ratio is not zero, validation amount is 0, train-test paradigm will be used.")
        tv_data, tv_labels = subject_data[:test_amount+val_amount], subject_label[:test_amount+val_amount] # test and validate
        train_data, train_labels = subject_data[test_amount+val_amount:], subject_label[test_amount+val_amount:]

        train_data, tv_data = self.preprocess_function(train_data, tv_data)

        test_data, test_labels = tv_data[:test_amount], tv_labels[:test_amount]
        val_data, val_labels = None, None
        if val_amount >= 1:
            val_data, val_labels = subject_data[test_amount:], subject_label[test_amount:]

        train_data, train_labels = segment(train_data, train_labels, self.segment_window)
        test_data, test_labels = segment(test_data, test_labels, self.segment_window)
        val_data, val_labels = segment(val_data, val_labels, self.segment_window)

        adjuster = None
        if self.encode:
            adjuster = BSASetEncoder()
        else:
            adjuster = SequenceStandardizer(self.mean, self.std,
                    self.avg_through_time)
        train_data = adjuster.fit_transform(train_data)
        test_data = adjuster.transform(test_data)
        if val_amount >= 1:
            val_data = adjuster.transform(val_data)

        train_set = {"data": train_data, "labels": train_labels}
        test_set = {"data": test_data, "labels": test_labels}
        val_set = {"data": test_data, "labels": test_labels}

        self.subject_idx += 1

        return train_set, test_set, val_set

def top_testbench(
        configured_iter,
        model_class, # Should not be the instance
        train_batch_size = 128,
        epochs =20,
        lr = 1e-5,
        ensemble = False):
    '''
    ensemble: if True, do the trial level result ensemble.
    '''

    segment_length = configured_iter.segment_window
    ensemble_amount = 1
    if ensemble:
        ensemble_amount = configured_iter.task_fEEG.shape[-1] // segment_length

    accuracy_list, record_list, confusion_matrix_list = [], [], []
    for i, (train_set, test_set, validate_set) in enumerate(configured_iter):
        print(f"###################### Fold {i+1} ######################")
        train_data = torch.from_numpy(train_set["data"]).to(dtype=torch.float32)
        train_labels = torch.from_numpy(train_set["labels"]).to(dtype=torch.long)
        test_data = torch.from_numpy(test_set["data"]).to(dtype=torch.float32)
        test_labels = torch.from_numpy(test_set["labels"]).to(dtype=torch.long)
        validate_data = torch.from_numpy(validate_set["data"]).to(dtype=torch.float32)
        validate_labels = torch.from_numpy(validate_set["labels"]).to(dtype=torch.long)

        train_ds = StaticSet(train_data, train_labels) # ds means dataset
        test_ds = StaticSet(test_data, test_labels)
        validate_ds = StaticSet(validate_data, validate_labels)

        train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=train_batch_size,
                shuffle=True, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=512, shuffle=False)
        validate_loader = torch.utils.data.DataLoader(
                validate_ds, batch_size=512, shuffle=False)

        model = model_class()
        if torch.cuda.is_available():
            model = model.cuda()

        trainer = NetworkTrainer(
                model, train_loader, test_loader,
                validate_loader = validate_loader,
                ensemble_amount = ensemble_amount,
                trainer_idx = i+1,
                lr = lr)
        test_segment_result, test_ensemble_result, test_confusion_matrix = trainer.execute(epochs)
        test_result = test_segment_result if test_ensemble_result is None else test_ensemble_result
        test_accuracy = test_result["accuracy"]
        accuracy_list.append(test_accuracy)
        record_list.append(test_result)
        confusion_matrix_list.append(test_confusion_matrix)

        print(f"Fold {i+1} Network accuracy: {test_accuracy}")

    return np.array(accuracy_list), record_list, confusion_matrix_list
