'''
Classical Reservoir tuned according to Lyapunov exponent.
'''

import numpy as np
import LSM_lib

if __name__ == "__main__":
    fdeap_file = "/home/featurize/fDEAP.h5"

    fEEG, labels = LSM_lib.EEG.load_DEAPh5(
            fdeap_file,
            "valence", # or "arousal"
            selected_electrodes = LSM_lib.EEG.DEAP_14_electrode_indices
            )
    
    window = 200

    is_iter = LSM_lib.training_utils.ISIter(
            fEEG, labels,
            LSM_lib.EEG.preprocess_as_FES, # or use lib.EEG.no_preprocess
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = window,
            encode = False,
            mean = 0,
            std = 1,
            avg_through_time = False
            )

    model = LSM_lib.models.EEGNetBaseline(window, band_amount=4, category_num=3) 

    _, record, _ = LSM_lib.training_utils.top_testbench(
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

    LSM_lib.training_utils.output_avg_se(record)
