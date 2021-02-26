import os
import numpy as np


def load_katakana_dataset():
    folder = r"/Users/samrullo/Documents/learning/data_science/jlda/skillup_ai/skillup_ai_material/DAY1/DAY1_vr6_0_0/4_kadai/1_data"
    data_filename = r"train_data.npy"
    labels_filename = r"train_label.npy"

    # data has the shape (3000,1,28,28) and labels has the shape (3000,15)
    data = np.load(os.path.join(folder, data_filename))
    labels = np.load(os.path.join(folder, labels_filename))
    return data, labels


def get_katakana_labels_dict():
    dic_katakana = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4, "ka": 5, "ki": 6, "ku": 7, "ke": 8, "ko": 9, "sa": 10,
                    "si": 11, "su": 12, "se": 13, "so": 14}
    return dic_katakana


def onehot_to_str(label):
    """
    ワンホットベクトル形式のラベルをカタカナ文字に変換する
    """
    dic_katakana = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4, "ka": 5, "ki": 6, "ku": 7, "ke": 8, "ko": 9, "sa": 10,
                    "si": 11, "su": 12, "se": 13, "so": 14}
    label_int = np.argmax(label)
    for key, value in dic_katakana.items():
        if value == label_int:
            return key
