import torch
import scipy.io as sio
from typing import *

class DESet(torch.utils.data.Dataset):
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data.cuda() if torch.cuda.is_available() else data
        self.targets = targets.cuda() if torch.cuda.is_available() else targets
    def __getitem__(self,idx):
        return self.data[idx], self.targets[idx]
    def __len__(self):
        return len(self.targets)

def get_loader(folder: str, label_type: str = "valence") -> Tuple[torch.utils.data.DataLoader]:
    '''
    Folder example: "/home/DE"
    label_type: "valence" or "arousal"
    Output: (train_loader, validate_loader)
    '''
    datasets = dict()
    ds_types = ["train", "validate"]
    for ds_type in ds_types:
        mat = sio.loadmat("{}/{}_ds.mat".format(folder, ds_type))
        data = torch.from_numpy(mat["data"]).to(dtype=torch.float32)
        labels = torch.from_numpy(mat["labels"]).to(dtype=torch.long)
        if label_type=="valence":
            labels = labels[:, 0]
        else:
            labels = labels[:, 1]
        datasets[ds_type] = DESet(data, labels)
    train_loader = torch.utils.data.DataLoader(datasets["train"], batch_size=32)
    validate_loader = torch.utils.data.DataLoader(datasets["validate"], batch_size=1080)
    return train_loader, validate_loader
