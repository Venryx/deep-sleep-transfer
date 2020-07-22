# pylint: disable=import-error, no-name-in-module

import os
import pickle
import sys
import time
import uuid as _uuid

# from data.Experiment import Experiment
from util.input_path_utils import get_src_parent_dir, get_dataset_path


def create_uuid(create: bool = True) -> str:
    """
    :param create: is a bool to make creation of the uuid optional
        and use 577000099446A instead
    :return: a uuid


    Description
    ----
    This method creates a pseudo uuid. this helps to identify objects

    Params
    ----

    """
    uuid = ""
    if create:
        uuid = _uuid.uuid4()
    else:
        uuid = "577000099446A"
        # print("Default_UUID: " + uuid)
    return uuid


def simple_obj_dump(subject, save_path: str):
    """
    :param save_path: path where the object should be stored.


    Description
    ----
    Method allows the storage of objects in files.

    Params
    ----

    """
    object_file = open(save_path + (subject.subject_name + '.pckl'), 'wb')
    pickle.dump(subject, object_file)
    object_file.close()


def name_subject_dump(subject, save_name: str, save_path: str):
    """
    :param save_name: path where the object should be stored.
    :param save_path: the name of the file.


    Description
    ----
    Method allows the storage of objects in files with particular name.

    Params
    ----

    """
    object_file = open(save_path + (save_name + '.pckl'), 'wb')
    pickle.dump(subject, object_file)
    object_file.close()


def load_pickle_to_obj(save_name: str, save_path: str) -> object:
    """
    Args:
        save_name: name of the pickle file
        save_path:  path where the object is located
    Returns: Object of the stored type.

    Description
    ----
    Method allows the load a stored object with particular name.

    Params
    ----
    """
    file_path = f"{save_path}{save_name}.pckl"

    try:
        with open(file_path, 'rb') as object_file:
            loaded_object = pickle.load(object_file)
    except pickle.PickleError:
        sys.exit(f"File \"{file_path}\" can not be loaded by pickle.load().")
    except FileNotFoundError:
        sys.exit(f"Can not find file at {file_path}!")
    return loaded_object


def load_pickle_to_experiment(save_name: str, save_path: str):
    loaded_object = load_pickle_to_obj(save_name, save_path)
    return loaded_object


def is_file_in(path: str, file_name: str) -> bool:
    """
    :param path: path in which to search
    :param file_name: which file or substring to search for
    :return: True if file found in path


    Description
    ----
    Checks whether file or substring is found in the path.

    Params
    ----

    """
    list_files = os.listdir(path)
    for element in list_files:
        if element.find(file_name) != -1:
            return True


def is_dir_in(path: str, dir_name: str) -> bool:
    """
    :param path: path in which to search
    :param dir_name: which file or substring to search for
    :return: True if file found in path


    Description
    ----
    Checks whether file or substring is found in the path.

    Params
    ----

    """
    return_bool = os.path.isdir(f"{path}{dir_name}/")
    #print(f"{path}{dir_name}")
    #print(return_bool)
    return return_bool


def check_and_make_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)       # @sp - bugfix
        except OSError:
            print(f"Folder did not exist and could not be created. "
                  f"({path})")


def get_processed_data_path(dataset_name, get_data_from_local_machine):
    """
    :return: path to the processed data saves


    Description
    ----
    Generates the path to the saved processed data inside your system.

    Note
    ----
    Only tested on local windows system!

    Params
    ----

    """
    dataset_path = get_dataset_path(dataset_name, get_data_from_local_machine)
    tail_path = "processed/"
    save_path = dataset_path + tail_path
    check_and_make_path(save_path)

    return save_path


def elapsed_time(func):
    def inner(*args, **kwargs):
        time_start = time.time()
        value = func(*args, **kwargs)
        time_end = time.time()
        print(f"elapsed time: {round((time_end - time_start), 1)} seconds\n")
        return value

    return inner


class DurationEstimator:

    def __init__(self):
        self.start_time = time.time()

    def estimate_duration(self, passed_units: int, remaining_units: int) -> str:
        """
        :param passed_units:  how many units have already been processed
        :param remaining_units: how many units remain
        :return: String e.g. "Estimated duration: 69 s"


        Description
        ----
        This is a tool which determines in a very simple way the expected
        duration of repetitive processes.

        Params
        ----

        """
        act_time = time.time()
        diff_time = act_time - self.start_time
        unit_time = diff_time / passed_units
        estimated_duration = f"Estimated duration: " \
                             f"{str(int(unit_time * remaining_units))} s"
        return estimated_duration


# shady workaround to suppress system outputs
def suppress_system_out():
    sys.stdout = open(os.devnull, 'w')
    # print("Suppressing SystemOut.")


# revert the workaround for the system outputs
def unsuppress_system_out():
    sys.stdout = sys.__stdout__
    # print("Stop suppressing SystemOut.")
