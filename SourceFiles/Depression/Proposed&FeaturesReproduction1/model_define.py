'''
Author: 
    Xiangnan Zhang: zhangxn@bit.edu.cn 
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025
Provides: 
    Definition of two-level fusion LSTM network

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''



import numpy as np

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

import h5py

import torch
from torch.utils.tensorboard import SummaryWriter

import shutil
from tqdm import tqdm
from typing import *

torch.manual_seed(42)
np.random.seed(42)

# Following: Metrics

def make_statistics(y_pred, y_true) -> dict:
    statis_dict = dict()
    statis_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    statis_dict["precision"] = metrics.precision_score(y_true, y_pred, pos_label=1)
    statis_dict["recall"] = metrics.recall_score(y_true, y_pred, pos_label=1)
    statis_dict["f1"] = metrics.f1_score(y_true, y_pred, pos_label=1)
    return statis_dict


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


# Following: machine learning method

def apply_classifier(train_pair: Tuple[np.ndarray], validate_pair, ensemble_amount, classifier="SVM") -> Tuple:
    '''
    Realization of SVM or KNN.
    classifier: SVM, KNN. Default SVM
    '''
    train_data, train_labels = train_pair
    validate_data, validate_labels = validate_pair

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    validate_data = scaler.transform(validate_data)
    
    if classifier=="SVM":
        classifier = SVC()
    elif classifier=="KNN":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier=="XGBoost":
        classifier = xgb.XGBClassifier(colsample_bytree=0.8, gamma=0.5, max_depth=7, min_child_weight=3, n_estimators=100)
    elif classifier=="RF": # Random Forest
        classifier = RandomForestClassifier(
            n_estimators=10,       # 树的数量
            max_depth=None,        # 不限深度（由数据决定）
            min_samples_split=2,   # 内部分裂的最小样本数
            min_samples_leaf=2,    # 叶子最小样本数
            max_features="sqrt",   # 每次分裂的特征子集
            random_state=42
        )
    elif classifier=="DT": # Decision Tree
        classifier = DecisionTreeClassifier(
            criterion='gini',         # 不纯度指标：'gini' or 'entropy'
            max_depth=None,              # 限制树深防止过拟合
            random_state=42           # 固定种子
        )
    else:
        raise ValueError("Classifier Setup error")
    classifier.fit(train_data, train_labels)
    pred = classifier.predict(validate_data)

    original_result = make_statistics(pred, validate_labels)
    ensemble_pred, ensemble_labels = make_ensemble(pred, validate_labels, ensemble_amount, None, "category")
    ensemble_result = make_statistics(ensemble_pred, ensemble_labels)
    return original_result, ensemble_result


# Following: Loader Define

def time_warping(data: torch.Tensor, ratio: tuple = (0.1,0.5)) -> torch.Tensor:
    B, C, T = data.shape
    min_len, max_len = int(T * ratio[0]), int(T * ratio[1])
    warped_len = torch.randint(min_len, max_len, (1,)).item()

    # 随机选择区间并分割
    start = torch.randint(0, T - warped_len, (1,)).item()
    end = start + warped_len
    mid = (start + end) // 2
    left, right = data[..., start:mid], data[..., mid:end]

    # 随机生成 pivot 点
    margin = int(warped_len * 0.3)
    pivot = int(torch.normal(0.5, 0.3, (1,)) * warped_len) + start
    pivot = max(min(pivot, end - margin), start + margin)

    left_warped = torch.nn.functional.interpolate(left, size=pivot-start, mode='linear', align_corners=True)
    right_warped = torch.nn.functional.interpolate(right, size=end-pivot, mode='linear', align_corners=True)

    # 拼接并保持原始长度
    return torch.cat([data[..., :start], left_warped, right_warped, data[..., end:]], dim=-1)


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) < 0.2:
        noise = torch.randn_like(x) * 0.5
    if torch.rand(1) < 0.2:
        x = x.flip(dims=[-1])
    if torch.rand(1) < 0.2:
        x = time_warping(x, (0.1, 0.3))
    return x


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


def get_loader(data_dir: str) -> Tuple[torch.utils.data.DataLoader]:
    '''
    Output: (train_loader, validate_loader)
    '''
    
    with h5py.File(data_dir, "r") as file:
        train_data = file["Train/data"][:]
        train_labels = file["Train/labels"][:]
        validate_data = file["Validate/data"][:]
        validate_labels = file["Validate/labels"][:]

    get_pos_ratio = lambda labels: labels.sum()/labels.shape[0]
    validate_pos_ratio = get_pos_ratio(validate_labels)
    train_pos_ratio = get_pos_ratio(train_labels)
    print("Positive ratio in validate_labels: {}, in train_labels: {}".format(validate_pos_ratio, train_pos_ratio))

    train_data = torch.from_numpy(train_data).to(dtype=torch.float32)
    train_labels = torch.from_numpy(train_labels).to(dtype=torch.long)
    validate_data = torch.from_numpy(validate_data).to(dtype=torch.float32)
    validate_labels = torch.from_numpy(validate_labels).to(dtype=torch.long)

    train_ds = StaticSet(train_data, train_labels, True)
    validate_ds = StaticSet(validate_data, validate_labels, False)
    print("Length of validate set: {}, train set: {}".format(len(validate_ds), len(train_ds)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=72, shuffle=True, num_workers=4)
    validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=512, shuffle=False)

    return train_loader, validate_loader


# Following: Trainer define

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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-2)
        weight_vec = torch.tensor([positive_ratio, 1-positive_ratio])
        if torch.cuda.is_available():
            weight_vec = weight_vec.cuda()
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

            validate_result, ensemble_result, confusion_matrix, attention_score_dict = self.validate(True)
            for key in validate_result.keys():
                self.log.add_scalar("validate/{}".format(key), validate_result[key], epoch+1)
            for key in ensemble_result.keys():
                self.log.add_scalar("ensemble/{}".format(key), ensemble_result[key], epoch+1)
            print("Validate loss: {}, validate accuracy: {}, ensemble accuracy: {}, F1: {}.".format(
                validate_result["loss"], validate_result["accuracy"], ensemble_result["accuracy"], ensemble_result["f1"]))
       
        print("Training Finished.")
        return ensemble_result["accuracy"], ensemble_result, confusion_matrix, attention_score_dict

# Following: Network Define

class Stream(torch.nn.Module):

    def __init__(self, original_channels, out_dim):
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
                input_size = 1,
                hidden_size = out_dim,
                num_layers = 1,
                batch_first = True
                )
        self.norm = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        '''
        x: [batch_size, 4, 125], i.e. data from a channel
        return: [batch_size, 64]
        '''
        x = x.transpose(-1, -2)
        _, (hn, cn) = self.lstm(x)
        x = hn[-1]
        x = self.norm(x)
        return x


class FeatureExtractor(torch.nn.Module):

    def __init__(self, feature_dim, band_amount=4):
        super().__init__()
        self.band_stream_pool = torch.nn.ModuleList(
                Stream(1, feature_dim) for _ in range(band_amount)
                )
        self.SE = torch.nn.Sequential(
                torch.nn.Linear(band_amount*feature_dim, band_amount*feature_dim),
                torch.nn.GELU(),
                torch.nn.Linear(band_amount*feature_dim, band_amount*feature_dim),
                torch.nn.Sigmoid()
                )
        self.decrease = torch.nn.Sequential(
                torch.nn.Linear(band_amount*feature_dim, feature_dim),
                torch.nn.GELU()
                )
        self.attention_score = torch.ones(1)
        if torch.cuda.is_available():
            self.attention_score = self.attention_score.cuda()
        self.band_amount = band_amount

    def forward(self, x):
        '''
        In: [bs, 4, timestep] in order (theta, alpha, beta, gamma).
        Out: [bs, feature_dim]
        '''
        thread_out = []
        for idx in range(self.band_amount):
            out = self.band_stream_pool[idx](x[:, idx].unsqueeze(1))
            thread_out.append(out)
        total = torch.cat(thread_out, dim=1)
        score = self.SE(total)
        self.attention_score = score
        y = total * score
        y = self.decrease(y)
        return y

class DenseRes(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.extract = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(dim, dim),
                torch.nn.GELU(),
                torch.nn.Linear(dim, dim)
                )
        self.combine = torch.nn.Sequential(
                torch.nn.BatchNorm1d(dim),
                torch.nn.GELU()
                )
    def forward(self, x):
        y = self.extract(x)
        y = self.combine(x+y)
        return y

class MainNet(torch.nn.Module):
    '''
    The proposed Two-level Fusion LSTM Network.
    '''

    def __init__(self):
        super().__init__()
        diff_dim = 64
        common_dim = 64
        self.diff_mode_extractor = FeatureExtractor(diff_dim)
        self.common_mode_extractor = FeatureExtractor(common_dim)
        self.fusion = torch.nn.Sequential(
                DenseRes(diff_dim+common_dim),
                DenseRes(diff_dim+common_dim),
                DenseRes(diff_dim+common_dim)
                )
        self.classifier = torch.nn.Linear(diff_dim+common_dim, 2)

    def forward(self, x):
        '''
        In: [bs, 4, 2, 125] in order (theta, alpha, beta, gamma).
        Out: [bs, 2]
        '''
        diff_features = self.diff_mode_extractor(x[:, :, 0, :])
        common_features = self.common_mode_extractor(x[:, :, 1, :])
        features = torch.cat([diff_features, common_features], dim=1)
        features = self.fusion(features)
        y = self.classifier(features)
        return y

class MainNet_BandAblation(torch.nn.Module):

    def __init__(self, ablation_mode: str, ablation_band_idx: int):
        '''
        ablation_mode: "differential" or "common"
        ablation_band_idx: 0, 1, 2, 3
        '''
        super().__init__()
        diff_dim = 64
        common_dim = 64
        diff_band = 3 if ablation_mode=="differential" else 4
        common_band = 3 if ablation_mode=="common" else 4
        if diff_band==common_band:
            raise ValueError("ablation_mode should be differential or common")
        self.diff_mode_extractor = FeatureExtractor(diff_dim, diff_band)
        self.common_mode_extractor = FeatureExtractor(common_dim, common_band)
        self.fusion = torch.nn.Sequential(
                DenseRes(diff_dim+common_dim),
                DenseRes(diff_dim+common_dim),
                DenseRes(diff_dim+common_dim)
                )
        self.classifier = torch.nn.Linear(diff_dim+common_dim, 2)
        self.ablation_mode = ablation_mode
        self.ablation_band_idx = ablation_band_idx

    def forward(self, x):
        '''
        In: [bs, 4, 2, 125] in order (theta, alpha, beta, gamma).
        Out: [bs, 2]
        '''
        diff_band_mask = [True, True, True, True]
        common_band_mask = [True, True, True, True]
        if self.ablation_mode=="differential":
            diff_band_mask[self.ablation_band_idx] = False
        else:
            common_band_mask[self.ablation_band_idx] = False
        diff_features = self.diff_mode_extractor(x[:, diff_band_mask, 0, :])
        common_features = self.common_mode_extractor(x[:, common_band_mask, 1, :])
        features = torch.cat([diff_features, common_features], dim=1)
        features = self.fusion(features)
        y = self.classifier(features)
        return y

class MainNet_ModeAblation(torch.nn.Module):

    def __init__(self, ablation_mode: str):
        super().__init__()
        dim = 64
        dim = 64
        self.diff_mode_extractor = None
        self.common_mode_extractor = None
        self.diff_mode_extractor = FeatureExtractor(dim)
        self.common_mode_extractor = FeatureExtractor(dim)
        if ablation_mode != "differential" and ablation_mode != "common":
            raise ValueError("ablation_mode should be differential or common")
        self.fusion = torch.nn.Sequential(
                DenseRes(dim),
                DenseRes(dim),
                DenseRes(dim)
                )
        self.classifier = torch.nn.Linear(dim, 2)
        self.ablation_mode = ablation_mode

    def forward(self, x):
        '''
        In: [bs, 4, 2, 125] in order (theta, alpha, beta, gamma).
        Out: [bs, 2]
        '''
        features = None
        if self.ablation_mode != "differential":
            features = self.diff_mode_extractor(x[:, :, 0, :])
        elif self.ablation_mode != "common":
            features = self.common_mode_extractor(x[:, :, 1, :])
        else:
            raise ValueError("ablation_mode should be differential or common")
        features = self.fusion(features)
        y = self.classifier(features)
        return y

if __name__ == "__main__":
    dataset_dir = "/home/xiangnan/E/EDoc/Research/SNN/Dataset/Depression/Original3Channel/DE_series.h5"
    train_loader, validate_loader = get_loader(dataset_dir)
    model = MainNet()
    trainer = EndToEndNetworkTrainer(model, train_loader, validate_loader, 1)
    trainer.train(10)
