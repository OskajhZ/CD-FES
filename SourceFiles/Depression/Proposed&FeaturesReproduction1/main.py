'''
Author: 
    Xiangnan Zhang: zhangxn@bit.edu.cn 
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025
Provides: 
    Entry of the proposed depression/normal classification system with both CD-FES and two-level fusion LSTM network

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''



import feature_define
import model_define
import numpy as np
import torch
import os
import h5py

def series_LSTM_cross_verification(spectrum_dir: str, **kwargs):
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
    data_type = kwargs.get("data_type", "CDFES")
    monitor_attention = kwargs.get("monitor_attention", True)

    TLFN_accuracy_list = []
    record_list = []
    confusion_matrix_list = []
    common_attention_list = []
    diff_attention_list = []
    folds = feature_define.CrossValidationIter(spectrum_dir, fold_amount, data_type)
    for i, (train_set, validate_set) in enumerate(folds):
        print("###################### Fold {} ######################".format(i+1))
        train_data = torch.from_numpy(train_set["data"]).to(dtype=torch.float32)
        train_labels = torch.from_numpy(train_set["labels"]).to(dtype=torch.long)
        validate_data = torch.from_numpy(validate_set["data"]).to(dtype=torch.float32)
        validate_labels = torch.from_numpy(validate_set["labels"]).to(dtype=torch.long)

        positive_ratio = (train_labels.sum() / len(train_labels)).item()

        train_ds = model_define.StaticSet(train_data, train_labels, True)
        validate_ds = model_define.StaticSet(validate_data, validate_labels, False)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)
        validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=512, shuffle=False)

        model = model_define.MainNet()
#        model = model_define.MainNet_ModeAblation("differential")
#        model = model_define.MainNet_BandAblation("differential", 3)

        trainer = model_define.EndToEndNetworkTrainer(
                model, train_loader, positive_ratio, validate_loader, validate_set["good_list"], ensemble_amount)
        TLFN_accuracy, record, confusion_matrix, attention_dict = trainer.train(epochs)
        TLFN_accuracy_list.append(TLFN_accuracy)
        record_list.append(record)
        confusion_matrix_list.append(confusion_matrix)
        if monitor_attention:
            common_attention_list.append(attention_dict["common"])
            diff_attention_list.append(attention_dict["differential"])
        print("############ Fold {} Network accuracy: {}".format(i+1, TLFN_accuracy))

    common_attention = np.concatenate(common_attention_list, axis=0).tolist() if monitor_attention else None
    diff_attention = np.concatenate(diff_attention_list, axis=0).tolist() if monitor_attention else None
        
    return np.array(TLFN_accuracy_list), record_list, confusion_matrix_list, common_attention, diff_attention

def CDFES_LSTM_main():
    spectrum_dir = "/home/xiangnan/E/EDoc/Research/情绪与抑郁识别/Dataset/Depression/Original3Channel/spectrum.h5"
    attention_score_file = "attention.h5"
    monitor_attention = True

    TLFN_accuracy_arr, record_list, confusion_matrix_list, common_attention, diff_attention = series_LSTM_cross_verification(
            spectrum_dir, 
            fold_amount = 10, epochs=20, ensemble_amount=28, train_batch_size=128, 
            data_type="CDFES",
            monitor_attention=monitor_attention)
    print(TLFN_accuracy_arr)
    print("Average accuracy: {}".format(TLFN_accuracy_arr.mean()))
    print("Full record:")
    print(record_list)
    print("Confusion Matrices:")
    print(confusion_matrix_list)
    
    if monitor_attention:
        with h5py.File(attention_score_file, "w") as root:
            root.create_dataset("common", data = common_attention)
            root.create_dataset("differential", data = diff_attention)
        print("Attention Scores Dump in \"{}\".".format(attention_score_file))

def moments_ML_cross_verification(spectrum_dir: str, **kwargs):
    '''
    Parameters in kwargs:
        ensemble_amount: default 28
        fold_amount: default 10
        data_type: default "CDFES"
        classifier: "SVM" or "KNN". Default "SVM"
    '''
    ensemble_amount = kwargs.get("ensemble_amount", 28)
    fold_amount = kwargs.get("fold_amount", 10)
    data_type = kwargs.get("data_type", "CDFES")
    classifier = kwargs.get("classifier", "SVM")

    accuracy_list = []
    record_list = []
    
    folds = feature_define.CrossValidationIter(spectrum_dir, fold_amount, data_type)
    for i, (train_set, validate_set) in enumerate(folds):
        print("###################### Fold {} ######################".format(i+1))
        train_set["data"] = feature_define.convert_series_to_moments(train_set["data"])
        validate_set["data"] = feature_define.convert_series_to_moments(validate_set["data"])
        train_pair = (train_set["data"], train_set["labels"])
        validate_pair = (validate_set["data"], validate_set["labels"])
        _, ensemble_result = model_define.apply_classifier(train_pair, validate_pair, ensemble_amount, classifier)
        accuracy_list.append(ensemble_result["accuracy"]) 
        record_list.append(ensemble_result)
        print("Accuracy: {}".format(ensemble_result["accuracy"]))

    accuracy_arr = np.array(accuracy_list)
    return accuracy_arr, record_list

def moments_ML_main(classifier="SVM"):
    spectrum_dir = "/home/xiangnan/E/EDoc/Research/情绪与抑郁识别/Dataset/Depression/Original3Channel/spectrum.h5"
    accuracy_arr, record_list = moments_ML_cross_verification(spectrum_dir, 
            ensemble_amount=28, fold_amount=10,
            data_type="CDFES", # Here to control CDFES or CDAED
            classifier=classifier)
    print("Accuracy Array:")
    print(accuracy_arr)
    
    print("Average Accuracy: {}".format(accuracy_arr.mean()))

    print("Results:")
    print(record_list)

if __name__ == "__main__":

# USE FOLLOWS TO TEST WITH CONVOLUTIONAL MACHINE LEARNING CLASSIFIERS
#    classifiers = ["SVM", "KNN", "DT", "RF", "XGBoost"]
#    for classifier in classifiers:
#        print("On {}".format(classifier))
#        moments_ML_main(classifier=classifier)

# USE FOLLOW TO TEST WITH TWO-LEVEL FUSION LSTM NETWORK
    CDFES_LSTM_main()
