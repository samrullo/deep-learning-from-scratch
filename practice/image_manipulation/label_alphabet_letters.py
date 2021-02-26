import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

folder = "/Users/samrullo/datasets/images/alphabet/letters"
img_files = os.listdir(folder)

for img_file in img_files:
    img = Image.open(os.path.join(folder, img_file))
    plt.imshow(img)
    plt.show()
    # label = input("Enter the letter label:")
    # plt.close()
    # new_img_filename = f"{label}_{img_file}"
    # os.rename(os.path.join(folder, img_file), os.path.join(folder, new_img_filename))
