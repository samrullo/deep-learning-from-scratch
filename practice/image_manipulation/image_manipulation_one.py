import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    pil_image = Image.fromarray(img)
    plt.imshow(pil_image)
    plt.gray()
    plt.show()
    return


def break_alphabet_image_into_letters(folder="/Users/samrullo/datasets/images/alphabet",
                                      original_file="alphabet_letters_00.jpg"):
    filepath = os.path.join(folder, original_file)
    img = Image.open(filepath)
    img_arr = np.array(img)
    print(f"img_arr shape : {img_arr.shape}")
    show_image(img_arr)

    folder_letters = os.path.join(folder, "letters")
    for fold_ in (folder, folder_letters):
        if not os.path.exists(fold_):
            os.makedirs(fold_)

    img_height = img_arr.shape[0]
    img_width = img_arr.shape[1]
    letter_img_size = img_width // 13

    for y in range(img_height):
        y_start = y * letter_img_size
        y_lim = (y + 1) * letter_img_size
        if y_lim > img_height:
            y_lim = img_height
        if y_start > img_height:
            y_start = img_height - letter_img_size
        if y_lim == img_height or y_start == img_height - letter_img_size:
            break
        for x in range(img_width):
            x_start = x * letter_img_size
            x_lim = (x + 1) * letter_img_size
            if x_lim > img_width:
                x_lim = img_width
            if x_start > img_width:
                x_start = img_width - letter_img_size
            letter_img = img_arr[y_start:y_lim, x_start:x_lim, :]
            if letter_img.shape[0] >= letter_img_size:
                letter_pil_img = Image.fromarray(letter_img)
                original_file_prefix = original_file.split(".")[0]
                letter_pil_img.save(os.path.join(folder_letters, f"{original_file_prefix}_letter_{y}_{x}.png"))
                print(f"saved letter {y},{x}")
            if x_lim == img_width or x_start == img_width - letter_img_size:
                break


folder = "/Users/samrullo/datasets/images/alphabet"
original_file = "alphabet_00.jpg"
break_alphabet_image_into_letters(folder, original_file)
