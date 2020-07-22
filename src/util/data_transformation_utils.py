import numpy as np
import pandas as pd
from keras.utils import to_categorical
from scipy.signal import butter, lfilter

# @sp - remove unused imports
from imblearn.over_sampling import SMOTE


def conv_arrays_to_df(train_x, train_y, params):
    n_steps = train_x.shape[2]  # number of samples per 30 seconds
    train_y_new = []
    for label in train_y:   #
        for channel in range(len(params.plx.get("ch_idx_list"))):
            train_y_new.append(label)
    train_y_new = np.array(train_y_new)
    # transform shape into (number of time periods, number of channels, 1)
    train_y = train_y_new.reshape(train_y.shape[0], len(params.plx.get("ch_idx_list")), 1)
    train = np.concatenate((train_x, train_y), axis=2)  # train --> train_x with labels
    labels = ([str(i) for i in range(0, n_steps)])
    labels.append("target")
    labels = np.array(labels)
    train = train.reshape(int(train.shape[0]) * int(train.shape[1]), train.shape[2])  # transform from 2 dimensions into 3 dimensions
    df = pd.DataFrame(train, columns=labels)
    return df


def convert_df_to_arrays(df_train_over, params):
    n_channels = len(params.plx.get("ch_idx_list")) # number of channels
    train_x = df_train_over.iloc[:, 0:-1].to_numpy()
    train_y = df_train_over.iloc[:, -1].to_numpy()
    train_x = train_x.reshape(int(int(train_x.shape[0]) / n_channels), n_channels, int(train_x.shape[1]))  # shape(number of time periods, number of channels, number of samples)
    train_y = train_y.reshape(int(int(train_y.shape[0]) / n_channels), n_channels)  # shape(number of time periods, number of channels)
    train_y = train_y[:, :1]
    return train_x, train_y

def perform_oversampling_smote(train_x_red, train_y_red, params):
    m, n, l = train_x_red.shape

    train_x_red = train_x_red.reshape((m, n*l))

    sampler = SMOTE()

    train_x_red, train_y_red = sampler.fit_resample(train_x_red, train_y_red)

    m, _ = train_x_red.shape

    return train_x_red.reshape((m, n, l)), train_y_red



def perform_oversampling(train_x_red, train_y_red, params):
    train_y_red = np.argmax(train_y_red, axis=1)
    df_train = conv_arrays_to_df(train_x_red, train_y_red, params)  # convert numpy arrays to pandas dataframe
    num_labels = len(params.plx.get('key_labels'))
    # df_class = {
    #     "0": df_train[df_train['target'] == 0],
    #     "1": df_train[df_train['target'] == 1],
    #     "2": df_train[df_train['target'] == 2],
    #     "3": df_train[df_train['target'] == 3],
    #     "4": df_train[df_train['target'] == 4]
    # }

    df_class = dict()
    for i in range(len(params.plx.get("key_labels"))):
        df_class[str(i)] = df_train[df_train['target'] == i]

    classes_value_counts = df_train.target.value_counts()
    print()
    series_per_label = df_train.shape[0] // classes_value_counts.size
    rest = df_train.shape[0] % classes_value_counts.size
    print("Class balance original\n")
    print(classes_value_counts)

    df_class_over = {}

    for label_idx in classes_value_counts.keys():
        df_class_over[str(int(label_idx))] = df_class[str(int(label_idx))].sample(
                series_per_label+rest, replace=True)
        rest = 0


    # while(label_idx < num_labels-1):
    #     if str(label_idx) in df_class.keys():
    #         df_class_over[str(label_idx)] = df_class[str(label_idx)].sample(
    #             series_per_label + rest, replace=True)
    #     label_idx += 1
    # if str(label_idx) in df_class.keys():
    #     df_class_over[str(label_idx)] = df_class[str(label_idx)].sample(
    #         series_per_label+rest, replace=True)
    # for ii, val in classes_value_counts.items():
    #     ii_str = str(int(ii))
    #     df_class_over[ii_str] = df_class[ii_str].sample(
    #         num_labels, replace=True)
    #     #df_class_over[ii_str] = df_class[ii_str].sample(n_samples_in_largest_class, replace=True)
    df_train_over = pd.concat(df_class_over, ignore_index=True, sort=False)
    print("Class balance after oversampling\n")
    print(df_train_over.target.value_counts())
    train_x_red, train_y_red = convert_df_to_arrays(df_train_over, params)  # convert pandas dataframe to numpy arrays
    train_y_red = to_categorical(train_y_red)
    return train_x_red, train_y_red



def select_data_subset(train_x, train_y, params):
    train_x_red = train_x[:params.last_interval, :, params.plx.get("ch_idx_list")]
    train_x_red = train_x_red.transpose(0, 2, 1)  # exchange columns with rows
    train_y_red = train_y[:params.last_interval, ]
    return train_x_red, train_y_red


def perform_feature_engineering(eng_kind, data_x):
    # >>> @sp - remove feature engineering
    if eng_kind != "raw":
        raise ValueError("Value \"" + eng_kind + "\" for parameter \"eng_kind\" is not supported")
    # <<< @sp - remove feature engineering


def reformat_labels_to_categorical(test_y_red, train_y_red=None, val_y_red=None):
    test_y_red = to_categorical(test_y_red)
    if train_y_red is not None:
        train_y_red = to_categorical(train_y_red)

        if val_y_red is not None:
            val_y_red = to_categorical(val_y_red)
            return test_y_red, train_y_red, val_y_red

        else:
            return test_y_red, train_y_red
    return test_y_red


def prepare_data_for_autoencoder(train_x, channel_idx, filter_bool=False):
    """

    :param train_x:
    :param filter_bool:
    :param i:
    :return:
    """

    train_x = train_x.transpose(1, 0, 2)
    tmp = train_x[channel_idx]# exchange columns with rows
    train_x = train_x[channel_idx].reshape(
        (train_x.shape[1], train_x.shape[2], 1))

    if filter_bool:
        train_x = butter_lowpass_filter(train_x)

    return train_x


def butter_lowpass_filter(data, cutoff=30, fs=200, order=6):
    normalized_cutoff = cutoff/(fs/2)
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    output = lfilter(b, a, data)
    return output