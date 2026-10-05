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

    def validate(self):
        print("Validating:")
        self.model.eval()
        avg_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []

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

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        avg_loss /= len(self.validate_loader.dataset)
        validate_result = make_statistics(all_preds.cpu().numpy(), all_labels.cpu().numpy())
        validate_result["loss"] = avg_loss

        all_logits = torch.cat(all_logits)
        ensemble_preds, ensemble_labels = make_ensemble(
                all_logits, all_labels, self.ensemble_amount, self.validate_good_list, "logits")
        ensemble_result = make_statistics(ensemble_preds, ensemble_labels)
        return validate_result, ensemble_result

    def train(self, epochs = 10):
        ensemble_result = None
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

            validate_result, ensemble_result = self.validate()
            for key in validate_result.keys():
                self.log.add_scalar("validate/{}".format(key), validate_result[key], epoch+1)
            for key in ensemble_result.keys():
                self.log.add_scalar("ensemble/{}".format(key), ensemble_result[key], epoch+1)
            print("Validate loss: {}, validate accuracy: {}, ensemble accuracy: {}, recall: {}.".format(
                validate_result["loss"], validate_result["accuracy"], ensemble_result["accuracy"], ensemble_result["recall"]))
       
        print("Training Finished.")
        return ensemble_result["accuracy"], ensemble_result

class CrossValidationIter():

    def __init__(self, dataset_dir: str, fold_num = 10):
        np.random.seed(36)
        with h5py.File(dataset_dir, "r") as root:
            self.total_data = root["data"][:] # [subjects, clips, 4 bands, 3 channels, 125]
            self.total_labels = root["labels"][:]

        shuffle_list = np.arange(len(self.total_data))
        np.random.shuffle(shuffle_list)
        self.total_data = self.total_data[shuffle_list]
        self.total_labels = self.total_labels[shuffle_list]

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
        data_shape = self.total_data.shape[2:]
        validate_set["data"] = self.total_data[validate_indices].reshape(-1, *data_shape)
        validate_set["labels"] = self.total_labels[validate_indices].reshape(-1)
        train_set["data"] = np.delete(self.total_data, validate_indices, axis=0).reshape(-1, *data_shape)
        train_set["labels"] = np.delete(self.total_labels, validate_indices, axis=0).reshape(-1)

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
            
            scaler = StandardScaler()
            train_set["data"] = scaler.fit_transform(train_set["data"].reshape(-1, 12)).reshape(-1, 4, 3)
            validate_set["data"] = scaler.transform(validate_set["data"].reshape(-1, 12)).reshape(-1, 4, 3)
            return train_set, validate_set

# Following: Network Define

class Attention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.F_trans = torch.nn.Linear(3,3)
        self.G_trans = torch.nn.Linear(3,3)
        self.H_trans = torch.nn.Linear(3,3)
        self.O_trans = torch.nn.Linear(3,3)
        self.gamma = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        '''
        x should firstly be converted into [bs, 3]
        '''
        F = self.F_trans(x).view(-1,3,1) # shape: [batch_size, width, channel]
        G = self.G_trans(x).view(-1,1,3)  # shape: [batch_size, channel, width]
        H = self.H_trans(x).view(-1,1,3) # shape: [batch_size, channel, width] 
        attention_matrix = torch.matmul(F,G)  # shape: [batch_size, width, width]
        attention_matrix = torch.nn.functional.softmax(attention_matrix, dim=-1).transpose(-2,-1)
        O = torch.matmul(H, attention_matrix) # shape: [batch_size, in_channel, width]
        O = O.squeeze()
        O = self.O_trans(O)
        y = self.gamma * O + x
        return y # shape: [batch_size, 3] 



class SFCSANet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        get_stream = lambda : torch.nn.Sequential(
                torch.nn.Linear(3,3),
                torch.nn.SELU(),
                Attention()) # out: [bs, 3]
        self.pcnn_theta = get_stream()
        self.pcnn_alpha = get_stream()
        self.pcnn_beta = get_stream()
        self.pcnn_gamma = get_stream()
        self.readout = torch.nn.Sequential(
                torch.nn.Linear(12, 1024),
                torch.nn.SELU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1024, 2))

    def forward(self, x):
        '''
        Return: Logits (without Softmax)
        In: [bs, 4, 3] in order (theta, alpha, beta, gamma).
        Out: []
        '''
        theta_out = self.pcnn_theta(x[:,0]) # shape: [bs, 256*32]
        alpha_out = self.pcnn_alpha(x[:,1])
        beta_out = self.pcnn_beta(x[:,2])
        gamma_out = self.pcnn_gamma(x[:,3])
        total = torch.cat((theta_out, alpha_out, beta_out, gamma_out), 1) # shape: [bs, 256*32*4]
        y = self.readout(total)
        return y


