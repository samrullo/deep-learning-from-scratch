import os
import Augmentor

folder_original = "/Users/samrullo/datasets/images/alphabet/original"
folder_augmented = "/Users/samrullo/datasets/images/alphabet/augmented"
for fold_ in (folder_augmented, folder_original):
    if not os.path.exists(fold_):
        os.makedirs(fold_)

p = Augmentor.Pipeline(folder_original)
# p.rotate(probability=0.9, max_left_rotation=20, max_right_rotation=20)
# p.zoom(probability=0.9, min_factor=1.5, max_factor=1.9)
# p.flip_left_right(probability=0.9)
p.flip_top_bottom(probability=0.9)
p.sample(71)
