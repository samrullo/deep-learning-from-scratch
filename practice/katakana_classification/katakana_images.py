import matplotlib.pyplot as plt
from PIL import Image


def show_katakana_image(img):
    img=img.reshape(28,28)
    pil_image = Image.fromarray(img)
    plt.imshow(pil_image, cmap='gray')
    plt.gray()
    plt.show()
    return
