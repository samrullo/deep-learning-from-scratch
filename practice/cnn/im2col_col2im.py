import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def im2col(input_data, FH, FW, stride=1, pad=0):
    N, C, H, W = input_data.shape

    # first apply padding to the image
    # since image has the shape of (N,C,H,W) we apply padding to 2nd and 3rd dimensions
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    # first with initialize col with shape (N,C,FH,FW,OH,OW)
    col = np.zeros((N, C, FH, FW, OH, OW))

    for y in range(FH):
        y_max = y + OH * stride
        for x in range(FW):
            x_max = x + OW * stride
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # eventually we return col with shape (N*OH*OW,C*FH*FW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(-1, C * FH * FW)
    return col


def col2im(col, input_shape, FH, FW, stride=1, pad=0, is_backward=True):
    N, C, H, W = input_shape
    OH = (H + 2 * pad - FH) // stride + 1
    OW = (W + 2 * pad - FW) // stride + 1

    # change col to have the shape (N,C,FH,FW,OH,OW)
    # originally col has the shape (N*OH*OW,C*FH*FW)
    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    # we initialize img as below, stride-1 is added to prevent errors due to image shrinking after im2col
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(FH):
        y_max = y + OH * stride
        for x in range(FW):
            x_max = x + OW * stride
            if is_backward:
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            else:
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == '__main__':
    folder = "/Users/samrullo/datasets/images/alphabet/letters"
    img_files = os.listdir(folder)
    img_file = img_files[np.random.choice(np.arange(len(img_files)), 1)[0]]
    img = Image.open(os.path.join(folder, img_file))
    img_arr = np.array(img)

    # bring Channel to the first dimension so that img has the shape (C,H,W)
    # then reshape it to (1,C,H,W)
    img_arr = img_arr.transpose(2, 0, 1)
    C, H, W = img_arr.shape
    img_arr = img_arr.reshape(1, C, H, W)

    print(f"{img_file} has the shape : {img_arr.shape}")
    FH, FW, stride, pad = 6, 6, 2, 1
    col = im2col(img_arr, FH, FW, stride=stride, pad=pad)
    print(f"resulting col shape from im2col : {col.shape}")
    rev_img = col2im(col, img_arr.shape, FH, FW, stride, pad, is_backward=False)
    print(f"img shape after col2im : {rev_img.shape}")
    diff = (rev_img - img_arr).sum()
    print(f"diff between img and rev_img : {diff}")
