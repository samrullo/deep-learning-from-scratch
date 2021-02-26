import os
import numpy as np
import pandas as pd


def load_params(folder, file_prefix):
    params = {}
    for file in os.listdir(folder):
        # remove file prefix and .npy
        param_name = file.replace(file_prefix, "")
        param_name = param_name.replace("_", "").replace(".npy", "")
        params[param_name] = np.load(os.path.join(folder, file))
    return params


my_params = load_params(f"/Users/samrullo/Documents/learning/data_science/jlda/nn_files/my_batch_norm",
                        "katakana_classification_v1_")
textbook_params = load_params(f"/Users/samrullo/Documents/learning/data_science/jlda/nn_files/textbook_batch_norm",
                              "katakana_classification_v1_")

print(f"param names : {my_params.keys()}")