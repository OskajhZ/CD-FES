import model_define
import numpy as np
import torch
import os


def fold_cross_verification(spectrum_dir: str, **kwargs):
    '''
    Parameters in kwargs:
        ensemble_amount: default 23
        fold_amount: default 5
        train_batch_size: default 32 
        epochs: default 5
    '''
    ensemble_amount = kwargs.get("ensemble_amount", 23)
    fold_amount = kwargs.get("fold_amount", 5)
    train_batch_size = kwargs.get("train_batch_size", 32)
    epochs = kwargs.get("epochs", 5)

    PCNN_accuracy_list = []
    record_list = []
    folds = model_define.CrossValidationIter(spectrum_dir, fold_amount)
    for i, (train_set, validate_set) in enumerate(folds):
        print("###################### Fold {} ######################".format(i+1))
        train_data = torch.from_numpy(train_set["data"]).to(dtype=torch.float32)
        train_labels = torch.from_numpy(train_set["labels"]).to(dtype=torch.long)
        validate_data = torch.from_numpy(validate_set["data"]).to(dtype=torch.float32)
        validate_labels = torch.from_numpy(validate_set["labels"]).to(dtype=torch.long)

        positive_ratio = (train_labels.sum() / len(train_labels)).item()

        train_ds = model_define.StaticSet(train_data, train_labels, False)
        validate_ds = model_define.StaticSet(validate_data, validate_labels, False)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=512, shuffle=False)

        model = model_define.SFCSANet()
        trainer = model_define.EndToEndNetworkTrainer(
                model, train_loader, positive_ratio, validate_loader, None, ensemble_amount)
        PCNN_accuracy, record = trainer.train(epochs)
        PCNN_accuracy_list.append(PCNN_accuracy)
        record_list.append(record)
        print("############ Fold {} Network accuracy: {}".format(i+1, PCNN_accuracy))
        
    return np.array(PCNN_accuracy_list), record_list
    
    
def ML_cross_verification(spectrum_dir: str, **kwargs):
    '''
    Parameters in kwargs:
        ensemble_amount: default 28
        fold_amount: default 10
        data_type: default "CDFES"
        classifier: "SVM" or "KNN". Default "SVM"
    '''
    ensemble_amount = kwargs.get("ensemble_amount", 28)
    fold_amount = kwargs.get("fold_amount", 10)
    classifier = kwargs.get("classifier", "SVM")

    accuracy_list = []
    record_list = []
    
    folds = model_define.CrossValidationIter(spectrum_dir, fold_amount)
    for i, (train_set, validate_set) in enumerate(folds):
        print("###################### Fold {} ######################".format(i+1))
        train_len = len(train_set["data"])
        validate_len = len(validate_set["data"])
        train_set["data"] = train_set["data"].reshape(train_len, -1)
        validate_set["data"] = validate_set["data"].reshape(validate_len, -1)
        train_pair = (train_set["data"], train_set["labels"])
        validate_pair = (validate_set["data"], validate_set["labels"])
        _, ensemble_result = model_define.apply_classifier(train_pair, validate_pair, ensemble_amount, classifier)
        accuracy_list.append(ensemble_result["accuracy"]) 
        record_list.append(ensemble_result)
        print("Accuracy: {}".format(ensemble_result["accuracy"]))

    accuracy_arr = np.array(accuracy_list)
    return accuracy_arr, record_list
    
def SFCSAN_main():
    spectrum_dir = "DE.h5"

    PCNN_accuracy_arr, record_list = fold_cross_verification(
            spectrum_dir, fold_amount = 10, epochs=20, ensemble_amount=28, train_batch_size=128)
    print(PCNN_accuracy_arr)
    print("Average accuracy: {}".format(PCNN_accuracy_arr.mean()))
    print("Full record:")
    print(record_list)
    

def ML_main(classifier="SVM"):
    spectrum_dir = "/home/xiangnan/E/EDoc/Research/情绪与抑郁识别/Dataset/Depression/Original3Channel/DE.h5"
    accuracy_arr, record_list = ML_cross_verification(spectrum_dir, 
            ensemble_amount=28, fold_amount=10,
            classifier=classifier)
    print("Accuracy Array:")
    print(accuracy_arr)
    
    print("Average Accuracy: {}".format(accuracy_arr.mean()))

    print("Results:")
    print(record_list)


if __name__ == "__main__":
    classifiers = ["SVM", "KNN", "DT", "RF", "XGBoost"]
    for classifier in classifiers:
        print("On {}".format(classifier))
        ML_main(classifier=classifier)

#    SFCSAN_main()
