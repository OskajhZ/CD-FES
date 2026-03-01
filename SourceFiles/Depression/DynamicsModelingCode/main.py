'''
Author: 
    Xiangnan Zhang: zhangxn@bit.edu.cn 
    (School of Future Technologies, Beijing Institute of Technology)
Year: 2025

The code is under the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition.
'''



import feature_define
import model_define
import numpy as np
import torch
import os
import h5py

def baseline_model_cross_verification(spectrum_dir: str, data_type, model_type, **kwargs):
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
    monitor_attention = kwargs.get("monitor_attention", True)

    NN_accuracy_list = []
    record_list = []
    confusion_matrix_list = []
    common_attention_list = []
    diff_attention_list = []

    folds = None
    if data_type in ["AED", "FES", "CDAED", "CDFES"]:
        folds = feature_define.SpectrumCrossValidationIter(spectrum_dir, fold_amount, data_type)
    elif data_type == "EEG":
        folds = feature_define.RawEEG_CrossValidationIter(spectrum_dir, fold_amount, subband=False)
    elif data_type == "subband":
        folds = feature_define.RawEEG_CrossValidationIter(spectrum_dir, fold_amount, subband=True)

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

        model = model_define.BaselineDynamicModel(train_data.shape[1], 64, out_dim=2, model_type=model_type)

        trainer = model_define.EndToEndNetworkTrainer(
                model, train_loader, positive_ratio, validate_loader, validate_set["good_list"], ensemble_amount)
        NN_accuracy, record, confusion_matrix, attention_dict = trainer.train(epochs)
        NN_accuracy_list.append(NN_accuracy)
        record_list.append(record)
        confusion_matrix_list.append(confusion_matrix)
        if monitor_attention:
            common_attention_list.append(attention_dict["common"])
            diff_attention_list.append(attention_dict["differential"])
        print("############ Fold {} Network accuracy: {}".format(i+1, NN_accuracy))

    common_attention = np.concatenate(common_attention_list, axis=0).tolist() if monitor_attention else None
    diff_attention = np.concatenate(diff_attention_list, axis=0).tolist() if monitor_attention else None
        
    return np.array(NN_accuracy_list), record_list, confusion_matrix_list, common_attention, diff_attention

def baseline_model_main(data_type, model_type):
    spectrum_dir = "/home/featurize/datasets/bf211208-479c-4c3e-ac65-e8f9864b4d1b/spectrum.h5"
    TLFN_accuracy_arr, record_list, confusion_matrix_list, common_attention, diff_attention = baseline_model_cross_verification(
            spectrum_dir, 
            data_type,
            model_type,
            fold_amount = 10, epochs=20, ensemble_amount=28, train_batch_size=128, 
            monitor_attention=False)
    print(TLFN_accuracy_arr)
    print("Average accuracy: {}".format(TLFN_accuracy_arr.mean()))
    print("Full record:")
    print(record_list)
    print("Confusion Matrices:")
    print(confusion_matrix_list)

if __name__ == "__main__":

#    data_type_list = ["subband", "AED", "FES", "CDAED", "CDFES"]
    data_type_list = ["EEG"]
    model_type_list = ["GRU"]

    for data_type in data_type_list:
        for model_type in model_type_list:
            print(f"!!!!!!!!!! {data_type} on {model_type} !!!!!!!!!!")
            baseline_model_main(data_type, model_type)

