DE_folder = "/home/featurize/work/Project/DE"
label_type = "valence"

from model_define import SFCSANet
from training_define import Trainer
from loader_define import get_loader

model = SFCSANet()
train_loader, test_loader = get_loader(DE_folder, label_type)
trainer = Trainer(model, train_loader, test_loader)

trainer.train(20)
