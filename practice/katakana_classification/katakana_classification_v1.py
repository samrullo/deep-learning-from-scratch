import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from practice.my_neural_networks.multilayer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from practice.multi_layer_with_batch_norm_option import MultiLayerNet
from practice.my_neural_networks.train_neural_networks import train_nn
from practice.katakana_classification.katakana_dataset import load_katakana_dataset
from practice.katakana_classification.katakana_images import show_katakana_image
from practice.katakana_classification.katakana_dataset import onehot_to_str
from practice.optimizers import RMSProp

katakana_data, katakana_labels = load_katakana_dataset()
print(f"katakana data shape : {katakana_labels.shape}")
print(f"katakana labels shape : {katakana_labels.shape}")

# let's analyse how many labels of each katakana we have
katakana_str_labels = [onehot_to_str(label) for label in katakana_labels]
kat_str_df = pd.DataFrame({'label': katakana_str_labels, 'label2': katakana_str_labels})
kat_grp_df = kat_str_df.groupby('label').count()
print(kat_grp_df)

data_size = katakana_data.shape[0]
train_size = round(data_size * 0.7)

# flatten katakana image samples to have one dimension. it becomes a vector with 784 elements
katakana_data = katakana_data.reshape(data_size, 28 * 28)

idx = np.arange(data_size)
np.random.seed(123)
np.random.shuffle(idx)
print(f"first 10 idx elements after shuffling : {idx[:10]}")

train_idx = idx[:train_size]
test_idx = idx[train_size:]
print(f"train idx length : {len(train_idx)}, test idx length : {len(test_idx)}")

X_train = katakana_data[train_idx, :]
y_train = katakana_labels[train_idx, :]

X_test = katakana_data[test_idx, :]
y_test = katakana_labels[test_idx, :]

print(f"X_train first element label:{onehot_to_str(y_train[0])}")
show_katakana_image(X_train[0])

# let's set up our multilayer neural network
input_size = X_train.shape[1]
output_size = y_train.shape[1]
katakana_nn = MultiLayerNetExtend(input_size, [100, 100, 100, 100, 100], output_size, 'relu', 0.01, 0,
                                  use_dropout=False, use_batchnorm=True)
# katakana_nn = MultiLayerNet(0.01, input_size, 100, output_size, 5, use_batch_norm=True)
optimizer = RMSProp(0.1, 0.9)
losses, train_accuracies, test_accuracies = train_nn(katakana_nn, X_train, y_train, X_test, y_test, optimizer, 100)

plt.figure()
plt.plot(train_accuracies)
plt.plot(test_accuracies)
plt.show()

folder = r"/Users/samrullo/Documents/learning/data_science/jlda/nn_files/textbook_batch_norm"
if not os.path.exists(folder):
    os.makedirs(folder)
file_prefix = f"katakana_classification_v1"

for param, param_arr in katakana_nn.params.items():
    filename = f"{file_prefix}_{param}.npy"
    np.save(os.path.join(folder, filename), param_arr)
    print(f"saved {param} to {filename}")
