import os


def get_src_parent_dir():
    current_directory = os.getcwd()
    src_parent_dir = current_directory[:current_directory.find("src")]
    return src_parent_dir


def get_dataset_path(dataset_name, get_data_from_local_machine):
    # >>> @sp - add support for winslow
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        # on winslow the processed data has to be stored in the workspace
        base_path = "/workspace/"
    # <<< @sp
    else:
        base_path = get_dataset_base_path(get_data_from_local_machine, verbose=False)

    return base_path + dataset_name + "/"


def get_raw_data_path(dataset_name, get_data_from_local_machine) -> str:
    """
    :param dataset_name: name of the dataset, must match the directory substring on the local
    :param get_data_from_local_machine: Load training data from a local folder instead.
    :return: path to the processed data saves


    Description
    ----
    Constructs the path to the saved training data.
    This can be located either on a local machine or on a dataset share.
    Both Windows and Linux systems are supported.

    Note
    ----
    Only tested on local windows system!

    Params
    ----

    """
    base_path = get_dataset_base_path(get_data_from_local_machine)
    file_path = base_path + dataset_path_mapping(dataset_name)
    return file_path


def get_dataset_base_path(get_data_from_local_machine, verbose=True):
    if get_data_from_local_machine:
        # >>> @sp - add support for winslow
        # if we are on winslow
        if 'WINSLOW_PIPELINE_NAME' in os.environ:
            if verbose:
                print("### INFO: Loading training data from the resource directory!")
            base_path = "/resources/medical/"
        # <<< @sp
        else:
            if verbose:
                print("### INFO: Loading training data from the local machine!")
            base_path = get_src_parent_dir() + "src/data/"      # @sp - put path to dataset here, e.g. "D:/"
    else:
        pass    # @sp - remove path to internal server
    return base_path


def dataset_path_mapping(dataset_name):
    switcher = {
        "physionet_challenge": "physionet_challenge/training/",
        "deep_sleep": "sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette/",
        "shhs1": "shhs/polysomnography/"    # @sp - add SHHS dataset
    }

    # >>> @sp - add support for winslow
    if dataset_name == "physionet_challenge" and 'WINSLOW_PIPELINE_NAME' in os.environ:
        dataset_path = "training/"
    # <<< @sp
    else:
        dataset_path = switcher.get(dataset_name, "Unknown choice of dataset_name")

    return dataset_path
