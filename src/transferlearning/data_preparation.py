# @sp: newly created

import os
import sys

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')

import params.polyaxon_parsing_iitnet_cnn_lstm as pp3
import random
import shutil

from data.process_data import process_data
from params.Params_Winslow import Params, winslow_params
from data.DataInterface import DataInterface as DataInt


def preprocess_physionet_data():
    """
    Only do preprocessing for all data to save storage space when training the model on
    large datasets (e.g. physionet). Save the data to the local disk (set as save_path in params).
    """

    print("Setup parameters ... ", end=" ")

    # get parameters
    params = Params()
    # get additional parameters for iitnet
    if params.plx.get('mdl_architecture') == "iitnet_cnn_bilstm":
        plx: dict = pp3.get_parameters()
        params.plx.update(plx)

    # adjust winslow parameters
    is_winslow = False
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        is_winslow = True
        winslow_params(params)

    params.plx['subject_batch'] = 1  # ! wichtig für IITNet

    print("done")
    print("\nBuild Data interpreter object: \n")

    # Get data
    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    total_subjects = params.plx.get('train_count') + params.plx.get('val_count') + params.plx.get('test_count')
    print("\nProcessing Data from", str(total_subjects), "subjects.")
    print("\nStart Data Processing ... ")

    # Process Data
    process_data(params, data_int, params.plx["data_count"])

    print("\n All Data processed.")


def preprocess_sleepedf_data():
    """
    Only do preprocessing for all data to save storage space when training the model on
    large datasets (e.g. physionet). Save the data to the local disk (set as save_path in params).
    """
    print("Setup parameters ... ", end=" ")

    # get parameters
    params = Params()
    # get additional parameters for iitnet
    if params.plx.get('mdl_architecture') == "iitnet_cnn_bilstm":
        plx: dict = pp3.get_parameters()
        params.plx.update(plx)

    is_winslow = False
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        is_winslow = True
        winslow_params(params)

    params.plx['subject_batch'] = 1  # ! wichtig für IITNet

    print("done")
    print("\nBuild Data interpreter object: \n")

    # Set in polyaxon-params: load=0, experiment-uuid=iitnet_0, get_raw_data_from_local_path=1,
    #                         data_already_processed=False, dataset_name=deep_sleep,
    #                         channel-types, channel-names, frequency, ch_idx_list
    # Set in preprocess_data_task_ssc: line 106 --> 7
    # input_path_utils: base_path = get_src_parent_dir() + "src/data/" (only local)
    if is_winslow:
        params.plx['save_path'] = '/output/sleep-edf-v1/sleep-cassette/processed/training/'
    else:
        params.plx['save_path'] = "D:/sleep-edf-v1/sleep-cassette/processed/training/"

    # Get data
    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    total_subjects = params.plx.get('train_count') + params.plx.get('val_count') + params.plx.get('test_count')
    print("\nProcessing Data from", str(total_subjects), "subjects.")
    print("\nStart Data Processing ... ")

    # Process Data
    process_data(params, data_int, params.plx["data_count"])

    print("\n All Data processed.")

    # Delete unnecessary files and separate test data
    cleanup_data(params=params, is_winslow=is_winslow)


def preprocess_shhs_data():
    """
    Only do preprocessing for all data to save storage space when training the model on
    large datasets (e.g. physionet). Save the data to the local disk (set as save_path in params).
    """
    print("Setup parameters ... ", end=" ")

    # get parameters
    params = Params()
    # get additional parameters for iitnet
    if params.plx.get('mdl_architecture') == "iitnet_cnn_bilstm":
        plx: dict = pp3.get_parameters()
        params.plx.update(plx)

    is_winslow = False
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        is_winslow = True
        winslow_params(params)

    params.plx['subject_batch'] = 1  # ! wichtig für IITNet

    print("done")
    print("\nBuild Data interpreter object: \n")

    # Set in polyaxon-params: load=0, experiment-uuid=iitnet_0, get_raw_data_from_local_path=1,
    #                         data_already_processed=False, dataset_name=shhs1,
    #                         channel-types, channel-names, frequency, ch_idx_list
    # Set in preprocess_data_task_ssc: line 106 --> 14/15/16
    # input_path_utils: base_path = "Z:/"

    # Get data
    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    total_subjects = params.plx.get('train_count') + params.plx.get('val_count') + params.plx.get('test_count')
    print("\nProcessing Data from", str(total_subjects), "subjects.")
    print("\nStart Data Processing ... ")

    # Process Data
    process_data(params, data_int, params.plx["data_count"])

    print("\n All Data processed.")

    # Delete unnecessary files and separate test data
    cleanup_data(params=params, is_winslow=is_winslow)

def cleanup_data(params, is_winslow):
    """
    Delete unnecessary large files, that are not needed for the model training.
    Separate Training and Testing Data.
    """
    # Delete unnecessary files
    folder = params.plx.get('save_path') + params.plx.get('experiment_uuid')
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    print("found", str(len(subfolders)), "folders in directory.")

    for folder in subfolders:
        x_file = folder + "/0_data_x_preprocessed.npy"
        print(x_file)
        if os.path.exists(x_file):
            os.remove(x_file)
            print("deleted")
        else:
            print("does not exist!")

        y_file = folder + "/0_data_y_preprocessed.npy"
        print(y_file)
        if os.path.exists(y_file):
            os.remove(y_file)
            print("deleted")
        else:
            print("does not exist!")

    # put randomly selected samples from training to test folder
    num_test = params.plx.get('test_count')
    test_subjects = random.sample(subfolders, num_test)
    print("Selected", str(len(test_subjects)), "test subjects.")

    if params.plx.get('dataset_name') == 'shhs1':  # only work with this dataset offline!
        dest_folder = 'D:/shhs1/processed/test/' + params.plx.get('experiment_uuid')
    elif params.plx.get('dataset_name') == 'deep_sleep':
        dest_folder = 'D:/sleep-edf-v1/sleep-cassette/processed/test/' + params.plx.get('experiment_uuid')
    elif params.plx.get('dataset_name') == 'physionet_challenge':
        dest_folder = 'D:/physionet_challenge/test/' + params.plx.get('experiment_uuid')

    for test_subject in test_subjects:
        print("Moving", test_subject, "to", dest_folder)
        shutil.move(test_subject, dest_folder)
        print("done.")

    print("Moved", str(len(test_subjects)), "subjects.")


if __name__ == "__main__":
    preprocess_shhs_data()
    # preprocess_sleepedf_data()
    # preprocess_physionet_data()
