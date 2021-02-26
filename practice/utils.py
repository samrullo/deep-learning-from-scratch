import numpy as np


def im2col(input_data, filter_h, filter_w, stride, pad):
    """
    Convert 4 dimensional image data into 2 dimensional matrix
    :param input_data: 4 dimensional image data (Number of images,Channels,Height,Width)
    :param filter_h: filter height
    :param filter_w: filter width
    :param stride: stride
    :param pad: pad
    :return: 2 dimensional matrix col
    """
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    N, C, input_h, input_w = img.shape
    output_h = int((input_h + 2 * pad - filter_h) / stride + 1)
    output_w = int((input_w + 2 * pad - filter_w) / stride + 1)

    col = np.zeros((N, C, filter_h, filter_w, output_h, output_w))

    for y in range(filter_h):
        y_max = y + output_h * stride
        for x in range(filter_w):
            x_max = x + output_w * stride
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * output_h * output_w, -1)
    return col

