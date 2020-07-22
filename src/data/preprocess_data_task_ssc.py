import numpy as np
from sklearn import preprocessing


def print_info(subject, info):
    print("Subject [ name: " + subject.subject_name + " uuid: " + str(
        subject.subject_uuid) + "]: " + info,
          end=" ... -> ")


def preprocess_subject(subject, encoder, key_labels, do_trim_wakephases=False):     # @sp - add trimming of wakephases

    print_info(subject, 'do trim')
    trim_raw_data(subject)
    print('done!')

    print_info(subject, "do cleanup annotations")
    clean_annotations(subject, key_labels)
    print('done!')

    print_info(subject, "do data transformation")
    transform_data(subject)
    print("done!")

    print_info(subject, "do annotation transformation")
    transform_annotations(subject, encoder, key_labels)
    print("done!")

    # >>> @sp - add trimming of wakephases
    if do_trim_wakephases:
        print_info(subject, "do trim wake phases")
        trim_wakephases(subject)
        print("done!")
    # <<< @sp


def trim_raw_data(subject):
    """
    :param subject: a subject e.g patient object (tr03-0005}

    Description
    ----
    This method is used to trim the raw data for each subject.

    Params
    ----

    """
    # a subject may have more than one unit (e.g. night) so we have to iterate
    # over them all
    for unit in subject.subject_raw_units:
        # frequency and sections together form our observation areas
        sample_frequency = subject.subject_sample_frequency
        sample_sections = subject.subject_sections
        sample_size = unit.shape[1]
        # we calculate the waste to make a reshape possible
        sample_waste = sample_size % (sample_frequency * sample_sections)
        # now we use the processed variable of the object to save the trimmed
        # matrix
        if sample_waste == 0:
            sample_trimmed = unit[:, :]
        else:
            sample_trimmed = unit[:, :-sample_waste]
        subject.append_processed_units(sample_trimmed)


def clean_annotations(subject, key_labels):
    """
    :param subject: a subject e.g patient object (tr03-0005}
    :param key_labels: a list of labels that should be kept e.g. ['W, 'N1',...]

    Description
    ----
    Creates a new dict from subject.subject_raw_units_labels, where any labels
    that aren't in key_labels are deleted. Saves the new dict to
    subject.subject_processed_units_labels

    """
    raw_labels = subject.subject_raw_units_labels
    processed_labels = [0]*len(raw_labels)
    for idx, raw_unit in enumerate(raw_labels):
        processed_labels[idx] = \
            {k: v for k, v in raw_unit.items() if v in key_labels}
    subject.subject_processed_units_labels = processed_labels


def transform_data(subject):
    """
    :param subject: a subject e.g patient object (tr03-0005}

    Description
    ----
    Reshaping the matrix of each subject e.g. patient to fulfill our conventions
    e.g. (857,6000,13)
    // (number of sections, section*frequency, number of channels)

    Params
    ----

    """
    # load frequency and sections to calc the right dimensions
    frequency = subject.subject_sample_frequency
    section_dur = subject.subject_sections
    sample_view = int(frequency * section_dur)

    for idx, unit in enumerate(subject.subject_processed_units):
        transposed = np.transpose(unit)
        reshaped = np.reshape(transposed, (-1, sample_view, 13))  # @sp - sleep-edf: 7, physionet: 13, shhs: 14/15/16
        subject.subject_processed_units[idx] = reshaped


def transform_annotations(subject, encoder,
                              key_labels: list):
    """
    :param subject: a subject e.g patient object (tr03-0005}
    :param encoder: a pre-fitted encoder
    :param key_labels: a list of important labels

    Description
    ----
    Also brings the classifications to the right measure. And encode it the
    right way.

    Params
    ----

    """
    window_length = subject.subject_sample_frequency*subject.subject_sections
    for idx, unit_labels in enumerate(subject.subject_processed_units_labels):
        labels, times = get_labels_and_times(unit_labels)
        window_idx = times//window_length
        end = subject.subject_processed_units[idx].shape[0]
        window_labels = fill_window_labels(labels, np.append(window_idx, end))
        encoded_labels = encode_labels(encoder, window_labels)
        subject.subject_processed_units_labels[idx] = encoded_labels


# >>> @sp - add trimming of wakephases
def trim_wakephases(subject):
    """
    Crop wake phases to 30 minutes before and after sleep (as described in paper by Back (2019))
    wake phase = label 4
    :param subject:
    """

    epoch_length_sec = subject.subject_sections
    num_epochs_keep = int(30 * 60 / epoch_length_sec)

    data_y = subject.subject_processed_units_labels[0]
    data_x = subject.subject_processed_units[0]

    # BEFORE SLEEPING
    # get number of wake epochs before sleeping
    counter_b = 0
    for label in data_y:
        if label == 4:
            counter_b += 1
        else:
            break
    # cut wake epochs
    if counter_b >= num_epochs_keep:
        cut_epochs_b = counter_b - num_epochs_keep
        data_y = data_y[cut_epochs_b:]
        data_x = data_x[cut_epochs_b:]

    # AFTER SLEEPING
    # get number of wake epochs after sleeping
    counter_a = 0
    flip_labels = data_y[::-1]
    for label in flip_labels:
        if label == 4:
            counter_a += 1
        else:
            break
    # cut wake epochs
    if counter_a > num_epochs_keep:
        cut_epochs_a = counter_a - num_epochs_keep
        data_y = data_y[:-cut_epochs_a]
        data_x = data_x[:-cut_epochs_a]

    subject.subject_processed_units = [data_x]
    subject.subject_processed_units_labels = [data_y]
# <<< @sp


def fill_window_labels(labels, change_indices):
    """

    :param labels:
    :param change_indices:
    :param length:
    :return:
    """
    start = change_indices[0]
    repetitions = np.diff(change_indices)
    repetitions[0] = repetitions[0] + start
    labels = np.repeat(labels, repetitions)
    return labels


def encode_labels(encoder, labels):
    """
    :param encoder: Encoder object with transform method
    :param labels: np.array of shape (l, 1) of labels to be encoded
    :return: np.array of shape (l, 1) of encoded labels
    """
    labels = np.asarray(labels)[:, np.newaxis]
    enc_labels = encoder.transform(labels)
    reshape_labels = enc_labels[:, np.newaxis]
    return reshape_labels


def get_labels_and_times(dict_):
    """
    Transform a dict containing times and labels into two np arrays
    :param dict_: A dict object where the keys are times and values are the
    labels of what happened at that time e.g. {72000:'W', ...}
    :return: np.array of labels and np.array of times
    """
    labels = np.asarray(list(dict_.items()))
    times = labels[:, 0].astype(int)
    labels = labels[:, 1]

    return labels, times

# @sp - remove unused methods
