import numpy as np

from data.normalizers import normalizer
# @sp - remove unnecessary imports
from util.auxiliary import load_pickle_to_experiment
from util.data_transformation_utils import perform_oversampling, \
    select_data_subset, reformat_labels_to_categorical


def process_data(params, data_int, count):
    """
    :param params: polyaxon params
    :param data_int: DataInterface object
    :param count: number of subjects to process

    Description
    ----
    if data_base is preprocessed: data is loaded from databasis "preprocessed",
    is prepared for training and saved under
    databasis "feature_eng_filename".

    Params
    ----
    """

    subject_count_per_stage = params.plx.get("subject_batch")
    data_base = params.plx.get("databasis")

    if data_base == "preprocessed":

        for processed_subjects in range(0, count, subject_count_per_stage):

            subjects_left = count - processed_subjects
            if subjects_left < subject_count_per_stage:
                subject_count_per_stage = subjects_left

            data_x, data_y = get_data(
                data_int=data_int,
                params=params,
                num_subjects=subject_count_per_stage)
            print(
                f"###### data_x shape: {data_x.shape} and data_y {data_y.shape}"
                f" with {subject_count_per_stage} subjects \n")

            data_x, data_y = \
                prepare_data_for_training(params, data_x=data_x, data_y=data_y)
            # save train_x to files to load it later again

            feature_eng_filename = params.plx.get("feature_eng_filename")

            data_int.store_particular_data(
                process_step_name=feature_eng_filename,
                data_x=data_x, data_y=data_y,
                data_count=subject_count_per_stage,
                offset=processed_subjects)


def get_data(data_int, params, num_subjects=None, start_val=0):
    """

    :param start_val:
    :param data_int:
    :param params:
    :param num_subjects:
    :return:

    Description
    ----
    Use Cases:
    - load raw data and preprocess it
    - load preprocessed data

    """

    if num_subjects is None:
        num_subjects = params.plx.get("subject_batch")

    if not params.plx.get("load"):
        preprocess_data(data_int, params, num_subjects)

    data_x, data_y = get_preprocessed_data(data_int, params,
                                           num_subjects,
                                           start_val)

    return data_x, data_y


def get_preprocessed_data(data_int, params, subject_count_per_stage,
                          start_val=0):
    # get subject data
    data_x = [0] * subject_count_per_stage
    data_y = [0] * subject_count_per_stage

    for idx in range(subject_count_per_stage):
        data_x[idx], data_y[idx] = data_int.get_next(
            process_step_name=params.plx["databasis"],
            uuid=params.plx["experiment_uuid"],
            data_count=params.plx["data_count"],
            start_val=start_val,
            params=params)      # @sp - add params

    data_x = np.concatenate(data_x)
    data_y = np.concatenate(data_y)

    return data_x, data_y


def preprocess_data(data_int, params, num_subjects):
    print("### INFO: Disabled file load. Starting from raw data!")

    # >>> @sp - add trimming of wakephases
    if params.plx['mdl_architecture'] == "iitnet_cnn_bilstm":
        trim_wakephases = params.plx['do_trim_wakephases']
    else:
        trim_wakephases = False
    # <<< @sp

    data_int.process_data_seq(
        get_data_from_local_machine=params.plx["get_raw_data_from_local_path"],
        dataset_name=params.plx["dataset_name"],
        data_count=num_subjects,
        subject_count_per_stage=params.plx["subject_batch"],
        frequency=params.plx["frequency"],
        sections=params.plx["sections"],
        channel_names=params.plx["channel_names"],
        channel_types=params.plx["channel_types"],
        process_step_name=params.plx["databasis"],
        permitted_overwrite=params.plx["permitted_overwrite"],
        do_trim_wakephases=trim_wakephases)     # @sp - add trimming of wakephases


def get_distribution_matrix(data_int, count):
    """

    :param data_int:
    :param count:
    :return: a list of distribution-matrices. Each matrix has the information of
    the class distribution of one subject

    Description
    ----
    loads the distribution-matrix for count subjects and attaches them together

    """

    tmp_matrix = data_int.get_distribution_matrix(0)
    matrix = np.array(tmp_matrix)

    for offset in range(1, count):
        tmp_matrix = data_int.get_distribution_matrix(offset)
        matrix = np.append(matrix, tmp_matrix)
        tmp_matrix = None

    return matrix


def prepare_data_for_training(params, data_x, data_y):
    """
    :param params: polyaxon params
    :param data_x: train values
    :param data_y: train classifications
    :return: data_x and data_y

    Description
    ----
    normalizes the data, calls function for feature-engineering, reformat the labels to categorical
    """
    data_x = normalizer(data_x)

    params.experiment.log_data_ref(data=data_x,
                                   data_name=params.plx.get('dataset_name'))

    # select subset of data by last interval
    data_x, data_y = select_data_subset(data_x, data_y, params)

    # @sp - remove unnecessary steps (for IITNet)

    # reformat labels to categorical data (e.g.
    # [0 3 1 4] -> [[1 0 0 0 0], [0 0 0 1 0], [0 1 0 0 0], [0 0 0 0 1]])
    data_y = reformat_labels_to_categorical(data_y)

    return data_x, data_y  # train_x, train_y, val_x, val_y


def prepare_data_for_extraction(params, data_x, data_y):
    """
    :param params: polyaxon params
    :param data_x: train values
    :param data_y: train classifications
    :return: data_x data_y


    Params
    ----
    """

    # log preprocessed data as data reference for polyaxon
    params.experiment.log_data_ref(data=data_x,
                                   data_name=params.plx.get('dataset_name'))

    # select subset of data by last interval
    # if params.plx.get("eng_kind") == "raw":
    data_x, data_y = select_data_subset(data_x, data_y, params)

    # perform oversampling of training data to get class balance
    data_x, data_y = perform_oversampling(data_x, data_y, params)

    return data_x, data_y


def process_data_for_evaluation(params, data_int, start_val=None, count=None):
    eng_kind = params.plx.get('eng_kind')
    data_base = params.plx.get("databasis")

    if start_val is None:
        start_val = params.plx.get("train_count") + params.plx.get("val_count")
    if count is None:
        count = params.plx.get("test_count")

    # subject_count_per_stage = params.plx.get("subject_batch")
    # data_cont_for_string = params.plx.get("train_count")
    # if data_cont_for_string - i < subject_count_per_stage:
    #     subject_count_per_stage = data_cont_for_string - i

    data_object_len = []

    subject_count_per_stage = params.plx.get("subject_batch")

    data_cont_for_string = count

    if params.plx.get("databasis") == "preprocessed":

        for i in range(0, count, subject_count_per_stage):
            if data_cont_for_string - i < subject_count_per_stage:
                subject_count_per_stage = data_cont_for_string - i

            data_x, data_y = get_data(data_int=data_int,
                                      params=params,
                                      start_val=0,
                                      num_subjects=subject_count_per_stage)
            print(
                f"###### data_x shape: {data_x.shape} and data_y {data_y.shape}"
                f" with {subject_count_per_stage} subjects \n")

            if data_base == "preprocessed":
                data_x, data_y = \
                    data_x, data_y = prepare_data_for_evaluation(params=params,
                                                                 data_x=data_x,
                                                                 data_y=data_y)
            # save train_x to files to load it later again

            feature_eng_filename = params.plx.get("feature_eng_filename")

            data_int.store_particular_data(
                process_step_name=feature_eng_filename,
                data_x=data_x, data_y=data_y,
                data_count=subject_count_per_stage,
                offset=i + start_val)

            data_object_len.append(data_x.shape[0])

    return data_object_len  # data_x, data_y#, val_x, val_y


def prepare_data_for_evaluation(params, data_x, data_y):
    data_x = normalizer(data_x)

    params.experiment.log_data_ref(data=data_x,
                                   data_name=params.plx.get('dataset_name'))

    # select subset of data by last interval
    data_x, data_y = select_data_subset(data_x, data_y, params)

    # @sp - remove unnecessary steps (for IITNet)

    # reformat labels to categorical data (e.g.
    # [0 3 1 4] -> [[1 0 0 0 0], [0 0 0 1 0], [0 1 0 0 0], [0 0 0 0 1]])
    data_y = reformat_labels_to_categorical(data_y)

    return data_x, data_y

# @sp - remove unused methods
