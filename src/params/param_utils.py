import math
import sys
import warnings
import tensorflow as tf

from util import auxiliary as aux

MAX_SUBJECT_COUNT_MAP = {
    "physionet_challenge": 994,
    "deep_sleep": 153,
    "shhs1": 5793       # @sp - add SHHS dataset
}

WARNING_SUBJECT_BATCH_VALUE = {
    "raw": 15,
    "morlet_tsinalis": 2,
    "ae": 15,
}


def set_params(params):
    set_test_count(params)
    set_data_count(params)
    set_data_count(params)
    set_save_path(params)
    handle_warnings(params)


def check_params(params):
    check_feature_eng_filename(params)
    check_subject_batch_size(params)


def set_save_path(params):
    if params['save_path'] is None:
        dataset_name = params.get('dataset_name')
        get_data_from_local_machine = params.get('get_raw_data_from_local_path')
        params['save_path'] = aux.get_processed_data_path(
            dataset_name, get_data_from_local_machine)


def set_test_count(params):
    if "train_test_ratio" in params:
        if params.get("train_test_ratio") is not None:
            train_test_ratio = params.get("train_test_ratio")
            train_count = params.get("train_count")
            params['test_count'] = math.ceil(train_count * train_test_ratio)
            print(f"\n\n### INFO: You set a train_test_ratio of "
                  f"\"{train_test_ratio}\", so your manually set entry's "
                  f"for test_subjects will be overwritten. "
                  f"Calculated: {train_count} for training and "
                  f"{params.get('test_count')} for testing.")
    else:
        print(f"\n\n### INFO: You set no train_test_ratio. "
              f"So your manually set entry for test_subjects will be "
              f"used.")


def set_data_count(params):
    params['data_count'] = params.get('train_count') + params.get('test_count') \
                           + params.get('val_count')
    if int(params.get('data_count')) > MAX_SUBJECT_COUNT_MAP.get(
            params.get('dataset_name')):
        sys.exit(f"Warning: The subject_count_per_stage is too high not "
                 f"enough data!")

# def set_permitted_overwrite(params):
#     # allow the program to overwrite generated data.
#     # Handle with care! DON'T CHANGE IT.
#     params['permitted_overwrite'] = 1

def check_feature_eng_filename(params):
    # do not allow to name your saved data == "preprocessed"
    if 'feature_eng_filename' in params:
        if params.get('feature_eng_filename') == "preprocessed":
            sys.exit("[ParameterError]: Please rename your feature_eng_filename"
                     " to something other than \"preprocessed\"!")
    else:
        print("There is no feature_eng_filename in polyaxon_parsing.")


def check_subject_batch_size(params):
    warning_value = 0
    try:
        warning_value = WARNING_SUBJECT_BATCH_VALUE[params["eng_kind"]]
    except KeyError:
        print("Can not find a warning value for your eng_kind in Params!")
    if int(params["subject_batch"]) > warning_value:
        print(f"Warning: The subject_count_per_stage is too high and "
              f"can lead to various memory issues!")


def handle_warnings(params):
    # not recommended but useful, filter tensorflow and python warnings
    if params.get('ignore_warnings'):
        print(f"\n\n### INFO: Ignoring warnings from Tensorflow and Python, "
              f"to hold the logfile clean.(Set the --ignore_warnings parameter "
              f"to False, to see them all!)\n")
        warnings.filterwarnings("ignore", category=Warning)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
