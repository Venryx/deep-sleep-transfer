# pylint: disable=import-error, no-name-in-module

import os
import sys
import time
import pickle

import numpy as np
from sklearn.preprocessing import LabelEncoder

import data.preprocess_data_task_ssc as data_preprocessor
from data import Subject
from data.deep_sleep import load_deep_sleep as loader_deep_sleep
from data.physionet_challenge import load_data_physionet as loader_physionet
from data.shhs import load_shhs as loader_shhs      # @sp - add SHHS dataset
from util import auxiliary as aux

# mapping the loaders with the dataset_name's
LOADER = {
    "physionet_challenge": loader_physionet,
    "deep_sleep": loader_deep_sleep,
    "shhs1": loader_shhs            # @sp - add SHHS dataset
}


class Experiment:

    def __init__(self,
                 key_labels: list = None,
                 frequency: int = 200,
                 sections: int = 30,
                 save_raw: bool = True,
                 do_save: bool = True,
                 save_path: str = None,
                 create_uuid: bool = False,
                 uuid: str = None):
        """
        Args:
            key_labels: Important labels to you. All the others will
            be removed from the data.
            frequency: E.g. Sample Frequency (e.g. 200Hz)
            sections: E.g. data will be divided into 30s sections
            save_raw: Boolean if you want to keep the raw data.
                If you choose "False" the raw data will be deleted immediately
                after creating the processed data
            do_save: Set the boolean "True" if you want to keep data at all.
            save_path: Set a save_path to load and save data.
            create_uuid: Should a pseudo uuid be created
            uuid: A uuid for the experiment

        Hint
        ----
        Please note that this is an experimental object and not suitable
        for production code!

        Description
        -----------
        A experiment simplifies the handling of the imported data.

        Params
        ------

        """
        if key_labels is None:
            key_labels = ["W", "N1", "N2", "N3", "R"]

        print("#### Create new experiment ####")
        self.creation_date = time.time()
        self.generate_uuid(create_uuid, uuid)
        self.key_labels = key_labels
        self.frequency = frequency
        self.sections = sections
        self.encoder = Experiment.__create_encoder(self)
        self.channel_count = None
        self.data_objects = dict()
        self.data_objects_concatenated = None
        self.data_annotations = None
        self.data_annotations_concatenated = None
        self.data_objects_list = []
        self.data_max_label = []
        self.save_raw = save_raw
        self.do_save = do_save
        self.save_path = f"{save_path}{self.uuid}/"
        print(f"#### Done create new experiment {self.uuid}####\n")

    def generate_uuid(self, create_uuid, uuid):
        """
        Args:
            create_uuid: Bool generate an uuid
            uuid: A given uuid or None

        Description
        -----------
        Generate the uuid or adopt the given one
        """
        if uuid is None:
            self.uuid = aux.create_uuid(create_uuid)
        else:
            self.uuid = uuid

    def __create_encoder(self):
        """
        Returns: Encoder

        Description
        -----------
        Generate and pre-fit an encoder.
        """
        print("Create new encoder", end=" ... -> ")
        encoder = LabelEncoder()
        encoder.fit(self.key_labels)
        print("done.")
        return encoder

    def load_seq(self, channel_names, channel_types, file_path, dataset_name,
                 frequency, sections, offset_seq=0, num_subjects=1, do_trim_wakephases=False):  # @sp - add trimming of wakephases
        """
        Args:
            dataset_name: Name of the used dataset
            num_subjects: Number of raw data to be processed.
            file_path: E.g path to the folder on the server
            frequency: Sample frequency e.g. 200Hz
            sections: E.g. divided into 30s sections
            channel_names: Name for each channel. Has a not None default
                inside the function.
            channel_types: Type for each channel. Has a not None default
                inside the function.
            offset_seq: Ignore the first x files.

        Description
        -----------
        Loads and processes the raw data. Trimming, transforming ... dropping.

        Params
        ------
        """

        aux.check_and_make_path(self.save_path)

        # load data
        for i in range(0, num_subjects):
            print(
                f"### Load data file {str(i + 1)} "
                f"of {str(num_subjects)}")

            subject = LOADER[dataset_name].load_data(
                file_path=file_path, frequency=frequency, sections=sections,
                channel_names=channel_names, channel_types=channel_types,
                offset=(i + offset_seq))

            # add subject to data_objects of the Experiment
            self.data_objects[subject.subject_name] = subject

            act_subject = self.data_objects[subject.subject_name]
            data_preprocessor.preprocess_subject(act_subject, self.encoder,
                                                 self.key_labels, do_trim_wakephases=do_trim_wakephases)    # @sp - add trimming of wakephases
            self.generate_concatenated(subject)
            unique, count = np.unique(subject.subject_processed_units_labels,
                                      return_counts=True)
            self.data_max_label.append(np.max(count))
            # clean memory
            unique = None
            count = None
            # self.plot_subject(subject.subject_name, 0, "processed")

            # generate distribution matrix
            distribution_matrix = self.get_distribution_matrix(
                act_subject.subject_processed_units_labels[0])
            self.save_distribution_matrix(distribution_matrix,
                                          self.save_path,
                                          act_subject.subject_name)
            print(act_subject)
            act_subject.dump(save_path=self.save_path, save_raw=self.save_raw)

        self.__create_file_lists()
        self.__dump_object()

    def generate_concatenated(self, subject: Subject):
        """

        Args:
            subject: A subject object

        Description
        -----------
        Concatenate all loaded subject data to two numpy arrays
        E.g. all nights.

        Params
        ------
        """
        # Todo: Maybe refactor and improve this. #code_duplicates
        self.data_objects_concatenated = None
        self.data_annotations_concatenated = None

        for element in self.data_objects.get(
                subject.subject_name).subject_processed_units:
            if self.data_objects_concatenated is None:
                self.data_objects_concatenated = element
            else:
                self.data_objects_concatenated = np.append(
                    self.data_objects_concatenated, element, axis=0)
            self.data_objects.get(subject.subject_name).create_indices()

        for element in self.data_objects.get(
                subject.subject_name).subject_processed_units_labels:
            if self.data_annotations_concatenated is None:
                self.data_annotations_concatenated = element
            else:
                self.data_annotations_concatenated = np.append(
                    self.data_annotations_concatenated, element, axis=0)

    def load_from_file(self, uuid: str = "577000099446A", data_count: int = 2,
                       offset: int = 0) -> []:
        """

        Args:
            uuid: Experiment uuid
            data_count: count of loaded data
            offset: Ignore the first x subjects

        Returns: Two numpy arrays from a subject.
        One time series data and one annotations.

        Description
        -----------
        Loading subject data from the local files.

        Params
        ------
        """
        print("### Load experiment from file ###")
        self.load_experiment(uuid)

        for subject_name in self.data_objects_list[offset:data_count + offset]:
            print(f"load subject: {subject_name}")
            placeholder_subject = Subject.Subject()
            placeholder_subject.load_from_file(subject_name, self.save_path)
            subject = placeholder_subject

            self.data_objects[subject.subject_name] = subject
            act_subject = self.data_objects.get(subject.subject_name)

            self.generate_concatenated(act_subject)
            act_subject.dump(save_path=self.save_path, save_raw=self.save_raw)

        print("### Done load experiment from file ###\n")

        return self.data_objects_concatenated, self.data_annotations_concatenated

    def load_stored_experiment(self, uuid, process_step_name, params=None):
        print(f"### Load stored user data: {process_step_name} ###")

        # >>> @sp - account for the possibility of separately preprocessed data
        # if the dataprocessing is done separately, there will be no pickle object.
        # set the parameters according to the params in polyaxon_parsing
        if params is not None:
            if params.plx.get("data_already_processed"):
                # self.uuid already specified
                # self.key_labels already specified
                # self.frequency already specified
                # self.sections already specified
                # self.encoder already specified
                self.channel_count = len(params.plx.get('ch_idx_list'))
                # load big arrays from npy files
                # self.data_objects_list already specified
                if self.data_objects is None:
                    self.data_objects = dict()
        # <<< @sp
        else:
            loaded_experiment = aux.load_pickle_to_experiment(uuid, self.save_path)
            try:
                self.uuid = loaded_experiment.uuid
                self.key_labels = loaded_experiment.key_labels
                self.frequency = loaded_experiment.frequency
                self.sections = loaded_experiment.sections
                self.encoder = loaded_experiment.encoder
                self.channel_count = loaded_experiment.channel_count
                # load big arrays from npy files
                self.data_objects_list = loaded_experiment.data_objects_list
                if self.data_objects is None:
                    self.data_objects = dict()

            except AttributeError:
                sys.exit(f"The loaded object of the experiment missing some "
                         f"attributes.")

    def __create_file_lists(self):
        """
        Description
        -----------

        Generates a list of included train and test subjects, which can be
        used to create the object again later.

        """
        for data_file in self.data_objects:
            if data_file not in self.data_objects_list:
                self.data_objects_list.append(data_file)

    def get_distribution_matrix(self, data):
        distribution_labels = self.key_labels
        distribution_dict = {}
        for label in distribution_labels:
            distribution_dict[label] = \
                list(self.encoder.inverse_transform(data)).count(label)
        print(f"Distribution Matrix: {distribution_dict}")
        return distribution_dict

    def save_distribution_matrix(self, distribution_matrix, save_path,
                                 subject_name):
        file_path = f"{save_path}{subject_name}"
        aux.check_and_make_path(file_path)
        file_path = file_path + "/distribution_matrix_preprocessed.pckl"
        with open(file_path, 'wb') as file:
            pickle.dump(distribution_matrix, file, pickle.HIGHEST_PROTOCOL)

    def __dump_object(self):
        """
        Description
        -----------

        Stores the experiment as an object and also splits the components
        into individual files.
        """

        recovery_for_annotations_concatenated = \
            self.data_annotations_concatenated
        recovery_for_objects_concatenated = \
            self.data_objects_concatenated
        recovery_for_annotations = self.data_annotations
        recovery_for_objects = self.data_objects

        self.data_objects_concatenated = None
        self.data_annotations_concatenated = None
        self.data_objects_concatenated = None
        self.data_annotations_concatenated = None

        aux.name_subject_dump(self, self.uuid, self.save_path)

        self.data_objects_concatenated = \
            recovery_for_objects_concatenated
        self.data_annotations_concatenated = \
            recovery_for_annotations_concatenated
        self.data_objects = recovery_for_objects
        self.data_annotations = recovery_for_annotations
        print("### Done save data ### \n")

    def get_indices(self) -> dict:
        """
        :return: A dict of indices for each subject (patient)

        Description
        ----
        Returns a dict in which the number of sections for each subject is
        returned. Extends self.aux_get_indices()

        Params
        ----

        """
        indices_data = Experiment.aux_get_indices(self, self.data_objects)
        return indices_data

    def aux_get_indices(self, object_list: dict) -> dict:
        """
        :return: a dict with name of the subject e.g. patient and the sum of
        sections on all units e.g. nights

        Description
        ----
        Retrieve start indices for each subject. Just a auxiliary function.

        Params
        ----

        """
        indices_dict = dict()
        for element in object_list:
            indices_dict[element] = self.data_objects.get(
                element).get_sum_indices()
        return indices_dict

    def store_data(self, process_step_name: str,
                   data_x: np.array, data_y: np.array,
                   data_count: int, offset: int = 0) -> None:
        """
         Args:
            process_step_name: The name under which the stored data should
            be saved and reloaded later.
            data_x: Data as array, which should be stored
            data_count: Number of subjects (patients)
            offset: Ignore the first x

        Description
        ----
        Stores arbitrary data under a specific name and within the
        associated subject.

        Hints
        ----
        Don't use "train_x, ... , test_y" or the subject name

        Params
        ----
        :param offset:
        :param data_count:
        :param data_x:
        :param process_step_name:
        :param data_y:

        """
        print(f"### Store stored user data: {process_step_name} ###")
        tmp_subject_list = list()
        for i in range(offset, data_count + offset):
            tmp_subject_list.append(self.data_objects_list[i])
        # Todo: does this work right ?
        indices_dict = self.aux_get_indices(tmp_subject_list)
        data_x_offset = 0
        for element in indices_dict:
            print(f"store subject {str(element)}")
            self.data_objects.get(element).store_data(
                process_step_name=process_step_name,
                data_x=
                data_x[data_x_offset:data_x_offset + indices_dict.get(element)],
                data_y=
                data_y[data_x_offset:data_x_offset + indices_dict.get(element)],
                save_path=self.save_path)
            data_x_offset += indices_dict.get(element)
        print(f"### Done store stored user data: {process_step_name} ###\n")

    def load_stored_data(self, uuid, process_step_name: str, num_subjects: int, params=None,    # @sp - add params
                         offset=0) -> np.array:
        """
        Args:
            uuid: The uuid of the experiment
            process_step_name: The name under which the stored data should
            be saved and reloaded later.
            num_subjects: Number of train subjects e.g. patients
            offset: Ignore the first x

        Description
        ----
        Load arbitrary data under a specific name and within the
        associated subject.

        Params
        ----

        """
        self.load_stored_experiment(uuid, process_step_name, params)    # @sp - add params
        self.load_subjects_to_experiment(num_subjects, offset)

        data_x = [0]*num_subjects
        data_y = [0]*num_subjects

        start = 0 + offset
        end = num_subjects + offset
        for idx, obj in enumerate(range(start, end)):
            self.generate_data_objects_list(obj)
            act_subject = self.data_objects[self.data_objects_list[obj]]

            print(f"\nload subject "
                  f"{act_subject.subject_name}")

            act_subject.load_stored_data(process_step_name, self.save_path)
            data_x[idx] = act_subject.get_stored_data(process_step_name)
            data_y[idx] = act_subject.get_stored_labels(process_step_name)

        data_x = np.concatenate(data_x)
        data_y = np.concatenate(data_y)

        print(f"### Done load stored user data: {process_step_name} ###\n")
        return data_x, data_y

    def load_subjects_to_experiment(self, num_subjects, offset=0):
        for subject_name in self.data_objects_list[
                            offset:num_subjects + offset]:
            placeholder_subject = Subject.Subject()
            placeholder_subject.load_from_file(subject_name, self.save_path)
            subject = placeholder_subject

            self.data_objects[subject.subject_name] = subject

    def generate_data_objects_list(self, index):
        if len(self.data_objects_list) < (index + 1):
            all_items = os.listdir(self.save_path)
            folders = []
            for item in all_items:
                if item.find(".") == -1:
                    folders.append(item)
            self.data_objects_list = folders

    # >>> @sp - account for the possibility of separately preprocessed data
    def recover_data_objectlist(self, objectlist):
        # recover the list of data objects, if data was preprocessed in a separate step
        self.data_objects_list = objectlist
    # <<< @sp

    def plot_subject(self, subject_name: str, subject_unit: int = 0,
                     mode: str = 'raw',
                     scaling: dict = None,
                     offset: int = 0):
        """
        Args:
            subject_name: the name of the subject e.g. patient tr03-0005
            subject_unit: the unit of the subject e.g. night 0
            mode: which data should be plotted e.g. raw or processed
            scaling: how to scale the different types of channels
                e.g. {'eeg': 20}
            offset: Ignore the first X seconds

        Description
        ----
        Plotting data of a subject e.g. patient with mne

        Params
        ----


        """
        if scaling is None:
            scaling = {'eeg': 20, 'misc': 200}

        if subject_name in self.data_objects:

            self.data_objects[subject_name].plot_mne(
                mode, scaling, subject_unit, offset)
        else:
            print("Object not in the train\\testset.\n")
    # Todo: Needed anymore ?
    # def split_train_test(self, data_x: [], data_y: [], train_count: int,
    #                      test_count: int) -> np.array:
    #     """
    #     :param data_x: Data as an array to be split.
    #     :param data_y: Data as an array to be split.
    #     :param train_count: Number of train subjects (patients)
    #     :param test_count: Number of test subjects (patients)
    #     :return: Four arrays divided by subjects
    #
    #     Description
    #     ----
    #     Split arrays by subjects.
    #
    #     Params
    #     ----
    #
    #     """
    #     try:
    #         train_indice = 0
    #         test_indice = 0
    #         indices = self.get_indices()
    #         indices_list = list()
    #         for element in indices:
    #             indices_list.append(element)
    #         for i in range(0, train_count):
    #             train_indice += indices.get(indices_list[i])
    #         for i in range(train_count, train_count + test_count):
    #             test_indice += indices.get(indices_list[i])
    #         train_x = data_x[:train_indice]
    #         test_x = data_x[train_indice:train_indice + test_indice]
    #         train_y = data_y[:train_indice]
    #         test_y = data_y[train_indice:train_indice + test_indice]
    #         return train_x, train_y, test_x, test_y
    #     except IndexError:
    #         sys.exit(f"[ParameterError]:Not enough data to split into "
    #                  f"{train_count} train and {test_count} test elements!")

    # def check_save_path(self):
    #     """
    #     Description
    #     -----------
    #     Check if the save path exits and create it if not.
    #
    #     """
    #     if os.path.isdir(self.save_path) is not True:
    #         try:
    #             os.mkdir(self.save_path)
    #         except OSError:
    #             print(f"Folder did not exist, but could not be
    #             created either."
    #                   f" ({self.save_path}")
