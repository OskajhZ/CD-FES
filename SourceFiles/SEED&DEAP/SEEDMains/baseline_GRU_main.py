'''
By Xiangnan Zhang, 2025
School of Future Technologies, Beijing Institute of Technology.
Version for the article: Frequency-Domain Entropy Sequence as a Dynamic EEG Representation for Depression Recognition
'''

import numpy as np
import lib

if __name__ == "__main__": # Use lib.EEG.dump_SEED() and .filter_SEED() to convert SEED into fseed.h5
    fseed_file = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"

    EEG_list, labels_list = [], []
    for session in ["session_1"]: # So as to 2 and 3
        EEG, labels = lib.EEG.load_SEEDh5(
                fseed_file,
                session = session,
                selected_electrodes = lib.EEG.SEED_14_electrode_indices
                )
        EEG_list.append(EEG)
        labels_list.append(labels)
    fEEG = np.concatenate(EEG_list, axis=1)
    labels = np.concatenate(labels_list, axis=1)

    is_iter = lib.training_utils.ISIter(
            fEEG, labels,
            lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = 200,
            encode = False,
            mean = 0,
            std = 1,
            avg_through_time = False
            )

    model = lib.models.GRUBaseline(band_amount=4, category_num=3, band_feature_dim = 128) 

    _, record, _ = lib.training_utils.top_testbench(
            is_iter,
            model,
            train_batch_size = 512,
            epochs = 300,
            reservoir_lr = 0,
            readout_lr = 1e-4,
            ensemble = False
            )

    print("Full Record:")
    print(record)

    lib.training_utils.output_avg_se(record)
