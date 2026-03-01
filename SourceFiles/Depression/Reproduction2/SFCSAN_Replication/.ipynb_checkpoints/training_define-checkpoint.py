import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn import metrics
from tqdm import tqdm



def make_statistics(y_pred, y_true) -> dict:
    statis_dict = dict()
    statis_dict["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    statis_dict["precision"] = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    statis_dict["recall"] = metrics.recall_score(y_true, y_pred, average="weighted")
    statis_dict["f1"] = metrics.f1_score(y_true, y_pred, average="weighted")
    return statis_dict


def append_dict(list_dict, item_dict) -> dict:
    for key in item_dict.keys():
        list_dict[key].append(item_dict[key])
    return list_dict
    


class Trainer():

    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, validate_loader):
        self.model = model
        if torch.cuda.is_available():
            print("Use CUDA")
            self.model.to("cuda")

        self.train_loader = train_loader
        self.validate_loader = validate_loader

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.dir = "logs/{}".format(current_time)
        self.log = SummaryWriter(log_dir = self.dir)
        print("log location: {}".format(self.dir))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        print("Optimizer configuration:")
        print(self.optimizer)

        input_samples, y = train_loader.dataset[0]
        input_samples = input_samples.unsqueeze(0)
        if torch.cuda.is_available():
            input_samples = input_samples.cuda()
        self.log.add_graph(self.model, input_samples)
        print("Trainer Initialization Finished.")

    def validate(self, idx):
        print("Validating:")
        self.model.eval()
        params = {"loss":[], "accuracy":[], "precision":[], "recall":[], "f1":[]}
        with torch.no_grad():
            for x,y in tqdm(self.validate_loader):
                pred_logits = self.model(x)
                loss = self.loss_fn(pred_logits, y)
                params["loss"].append(loss.cpu().item())
                pred = torch.argmax(pred_logits, dim=1)
                statis_dict = make_statistics(pred.cpu().numpy(), y.cpu().numpy())
                params = append_dict(params, statis_dict)
        for key in params.keys():
            param_array = np.array(params[key])
            param = param_array.mean().item()
            self.log.add_scalar("validate/{}".format(key), param, idx)
            params[key] = param # change of data type (from list to float)
        print("Validate loss: {}, validate accuracy: {}".format(params["loss"], params["accuracy"]))
    
    def train(self, epochs = 10):
        print("Base validation:")
        self.validate(0)
        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            loss_list = np.array([])
            accuracy_list = np.array([])
            self.model.train()
            for i,(x,y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                pred_logits = self.model(x)
                loss = self.loss_fn(pred_logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i % (len(self.train_loader)//3)==0 and i!=0) or i==len(self.train_loader)-1:
                    loss_list = np.append(loss_list, loss.detach().cpu().item())
                    pred = torch.argmax(pred_logits, dim=1)
                    accuracy = metrics.accuracy_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
                    accuracy_list = np.append(accuracy_list, accuracy)
                    torch.save(self.model.state_dict(), "{}/model_state_dict.pt".format(self.dir))
            loss = loss_list.mean().item()
            accuracy = accuracy_list.mean().item()
            self.log.add_scalar("train/loss", loss, epoch)
            self.log.add_scalar("train/accuracy", accuracy, epoch)
            print("Train loss: {}, train accuracy: {}".format(loss, accuracy))
            self.validate(epoch+1)
        print("Training Finished.")

