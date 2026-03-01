'''
Author: 
    Xiangnan Zhang: zhangxn@bit.edu.cn 
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025
Provides: 
    Reproduction of EEGNet

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import h5py
from sklearn import metrics

import shutil
from tqdm import tqdm
from typing import *

torch.manual_seed(42)

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

class CrossValidationIter():

    def __init__(self, dataset_dir: str, fold_num = 10):
        np.random.seed(42)
        with h5py.File(dataset_dir, "r") as root:
            EEG_group = root["OriginalEEG"]
            self.total_data = EEG_group["data"][:][..., :84*250]
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

def make_statistics(y_pred, y_true) -> dict:
    statis_dict = dict()
    statis_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    statis_dict["precision"] = metrics.precision_score(y_true, y_pred, pos_label=1)
    statis_dict["recall"] = metrics.recall_score(y_true, y_pred, pos_label=1)
    statis_dict["f1"] = metrics.f1_score(y_true, y_pred, pos_label=1)
    return statis_dict

class StaticSet(torch.utils.data.Dataset):

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, apply_augment=False):
        self.data = data
        self.targets = targets
        self.apply_augment = apply_augment
    def set_augment(self, mode: bool):
        self.apply_augment = mode
    def __getitem__(self,idx):
        single_data = self.data[idx]
        if self.apply_augment:
            single_data = augment(single_data)
        return single_data, self.targets[idx]
    def __len__(self):
        return len(self.targets)


class EndToEndNetworkTrainer():

    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, positive_ratio, validate_loader, validate_good_list, ensemble_amount = 3):
        self.model = model
        if torch.cuda.is_available():
            print("Use CUDA")
            self.model.to("cuda")

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.ensemble_amount = ensemble_amount
        self.validate_good_list = validate_good_list

        self.dir = "logs"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.log = SummaryWriter(log_dir = self.dir)
        print("log location: {}".format(self.dir))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        weight_vec = torch.tensor([positive_ratio, 1-positive_ratio]).cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05, weight=weight_vec)
        print("Optimizer configuration:")
        print(self.optimizer)

        input_samples, y = train_loader.dataset[0]
        input_samples = input_samples.unsqueeze(0)
        if torch.cuda.is_available():
            input_samples = input_samples.cuda()
        self.log.add_graph(self.model, input_samples)
        print("Trainer Initialization Finished.")

    def get_category(self, logits):
        return torch.argmax(logits, dim=1)

    def get_probability(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=1)
        positive_probs = probs[:, 1]
        return positive_probs

    def save(self):
        torch.save(self.model.state_dict(), "PCNN_state_dict.pt")

    def load(self):
        state_dict = torch.load("PCNN_state_dict.pt")
        self.model.load_state_dict(state_dict)

    def validate(self, with_attention_score=False):
        print("Validating:")
        self.model.eval()
        avg_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []
        common_attention_score_list = []
        differential_attention_score_list = []

        with torch.no_grad():
            for x,y in tqdm(self.validate_loader):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                pred_logits = self.model(x)
                loss = self.loss_fn(pred_logits, y)
                avg_loss += loss.cpu().item()*len(y)
                pred = self.get_category(pred_logits)

                all_preds.append(pred)
                all_labels.append(y)
                all_logits.append(pred_logits)

                if with_attention_score:
                    common_score = self.model.common_mode_extractor.attention_score.cpu().numpy()
                    diff_score = self.model.diff_mode_extractor.attention_score.cpu().numpy()
                    common_attention_score_list.append(common_score)
                    differential_attention_score_list.append(diff_score)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        avg_loss /= len(self.validate_loader.dataset)
        validate_result = make_statistics(all_preds.cpu().numpy(), all_labels.cpu().numpy())
        validate_result["loss"] = avg_loss

        all_logits = torch.cat(all_logits)
        ensemble_preds, ensemble_labels = make_ensemble(
                all_logits, all_labels, self.ensemble_amount, self.validate_good_list, "logits")
        ensemble_result = make_statistics(ensemble_preds, ensemble_labels)
        confusion_matrix = metrics.confusion_matrix(ensemble_labels, ensemble_preds)

        attention_score_dict = {"common": None, "differential": None}
        if with_attention_score:
            attention_score_dict["common"] = np.concatenate(common_attention_score_list, axis=0)
            attention_score_dict["differential"] = np.concatenate(differential_attention_score_list, axis=0)

        return validate_result, ensemble_result, confusion_matrix, attention_score_dict

    def train(self, epochs = 10):
        ensemble_result = None
        confusion_matrix = None
        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            loss_list = np.array([])
            accuracy_list = np.array([])
            self.model.train()
            for i,(x,y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                pred_logits = self.model(x)
                loss = self.loss_fn(pred_logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
            loss = loss.detach().cpu().item()
            pred = self.get_category(pred_logits)
            accuracy = metrics.accuracy_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            self.log.add_scalar("train/loss", loss, epoch+1)
            self.log.add_scalar("train/accuracy", accuracy, epoch+1)
            print("Train loss: {}, train accuracy: {}".format(loss, accuracy))

            validate_result, ensemble_result, confusion_matrix, _ = self.validate(False)
            for key in validate_result.keys():
                self.log.add_scalar("validate/{}".format(key), validate_result[key], epoch+1)
            for key in ensemble_result.keys():
                self.log.add_scalar("ensemble/{}".format(key), ensemble_result[key], epoch+1)
            print("Validate loss: {}, validate accuracy: {}, ensemble accuracy: {}, F1: {}.".format(
                validate_result["loss"], validate_result["accuracy"], ensemble_result["accuracy"], ensemble_result["f1"]))
       
        print("Training Finished.")
        return ensemble_result["accuracy"], ensemble_result, confusion_matrix


class EEGNet(nn.Module):
    """
    PyTorch EEGNet (v4) 实现
    论文: Lawhern et al., 2018, "EEGNet: a compact convolutional network..."
    输入: x ∈ [N, 1, C, T]
    关键超参:
      C: 通道数 (channels)
      T: 时间点数 (samples)
      F1: 第一层滤波器个数
      D:  深度可分离倍增系数
      F2: 第二块的输出通道数 (=F1*D)
      kernel_length: 第一层时间卷积核长度
      dropout: Dropout 概率
    """
    def __init__(
        self,
        num_classes: int,
        C: int,
        T: int,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.5,
        pool1: int = 4,
        pool2: int = 8,
        separable_kernel: int = 16,
    ):
        super().__init__()
        self.C = C
        self.T = T
        F2 = F1 * D

        # Block 1: Temporal Convolution -> Depthwise Spatial Convolution
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # depthwise spatial: 在通道维C上卷积，groups=F1 实现每个时间滤波器独立做空间卷积
        self.conv_depthwise_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(C, 1),
            groups=F1,               # depthwise
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(p=dropout)

        # Block 2: Separable Convolution (depthwise temporal + pointwise)
        # depthwise temporal
        self.conv_depthwise_temporal = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=F1 * D,          # depthwise
            bias=False
        )
        # pointwise 1x1
        self.conv_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(p=dropout)

        # 分类头：自适应池化到 1×1，再线性分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F2, num_classes, bias=True)
        )

        # 激活函数
        self.act = nn.ELU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_temporal(x)          # [N, F1, C, T]
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv_depthwise_spatial(x) # [N, F1*D, 1, T]
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)                  # [N, F1*D, 1, T/pool1]
        x = self.drop1(x)

        x = self.conv_depthwise_temporal(x)# [N, F1*D, 1, ·]
        x = self.conv_pointwise(x)         # [N, F2, 1, ·]
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.classifier(x)             # [N, num_classes]
        return x

def cross_verification(spectrum_dir: str, **kwargs): # EEG contains in spectrum.h5
    '''
    Parameters in kwargs:
        ensemble_amount: default 23
        fold_amount: default 5
        train_batch_size: default 32 
        epochs: default 5
        data_type: default "CDFES"
    '''
    ensemble_amount = kwargs.get("ensemble_amount", 23)
    fold_amount = kwargs.get("fold_amount", 5)
    train_batch_size = kwargs.get("train_batch_size", 32)
    epochs = kwargs.get("epochs", 5)

    accuracy_list = []
    record_list = []
    confusion_matrix_list = []
    folds = CrossValidationIter(spectrum_dir, fold_amount)
    for i, (train_set, validate_set) in enumerate(folds):
        print("###################### Fold {} ######################".format(i+1))
        train_data = torch.from_numpy(train_set["data"]).to(dtype=torch.float32)
        train_labels = torch.from_numpy(train_set["labels"]).to(dtype=torch.long)
        validate_data = torch.from_numpy(validate_set["data"]).to(dtype=torch.float32)
        validate_labels = torch.from_numpy(validate_set["labels"]).to(dtype=torch.long)

        positive_ratio = (train_labels.sum() / len(train_labels)).item()

        train_ds = StaticSet(train_data, train_labels, False)
        validate_ds = StaticSet(validate_data, validate_labels, False)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=512, shuffle=False)

        model = EEGNet(2, 3, 3*250)
        trainer = EndToEndNetworkTrainer(
                model, train_loader, positive_ratio, validate_loader, validate_set["good_list"], ensemble_amount)
        accuracy, record, confusion_matrix = trainer.train(epochs)
        accuracy_list.append(accuracy)
        record_list.append(record)
        confusion_matrix_list.append(confusion_matrix)
        print("############ Fold {} Network accuracy: {}".format(i+1, accuracy))

    return np.array(accuracy_list), record_list, confusion_matrix_list

def EEGNet_main():
    spectrum_dir = "spectrum.h5"

    accuracy_arr, record_list, confusion_matrix_list = cross_verification(
            spectrum_dir, 
            fold_amount = 10, epochs=20, ensemble_amount=28, train_batch_size=128) 
    print(accuracy_arr)
    print("Average accuracy: {}".format(accuracy_arr.mean()))
    print("Full record:")
    print(record_list)
    print("Confusion Matrices:")
    print(confusion_matrix_list)

if __name__ == "__main__":
    EEGNet_main()

