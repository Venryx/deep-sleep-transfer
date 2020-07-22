import numpy as np
from sklearn.preprocessing import minmax_scale


def normalizer(data_x):
    """
    :param data_x: the whole set of data for the sequence
    :return: data_x

    Description
    ----
    This normalizes the whole data for each channel.

    Params
    ----
    """
    placeholder = None
    data_shape = data_x.shape
    placeholder = np.empty(data_shape)
    for j in range(data_shape[0]):
        tmp_list = data_x[j]
        tmp_list_norm = minmax_scale(tmp_list, axis=0)
        placeholder[j] = np.array(tmp_list_norm)
    data_x = placeholder
    placeholder = None
    tmp_list = None
    tmp_list_norm = None
    return data_x
