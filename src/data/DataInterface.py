# pylint: disable=import-error, no-name-in-module

import os
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder as LabelEncoder

from data.Experiment import \
    Experiment as ExExOb
from util import auxiliary as af
from util.auxiliary import elapsed_time, load_pickle_to_obj
from util.input_path_utils import get_raw_data_path
from util.run_environment_utils import is_running_locally


class DataInterface:

    def __init__(self, save_path: str,
                 perform_save_raw: bool,
                 key_labels: list = None,
                 uuid: str = None):
        """
        Args:
            save_path: path in which the data is stored
            perform_save_raw: Load raw data from a local folder instead.
            key_labels: Important labels
            uuid: a specific uuid if needed

        Description
        ----
        This data interface should help to make the access to the data easier.

        Params
        ----

        """
        if key_labels is None:
            key_labels = ["W", "N1", "N2", "N3", "R"]

        self.experiment = ExExOb(save_path=save_path,
                                 save_raw=perform_save_raw,
                                 key_labels=key_labels,
                                 uuid=uuid)
        self.offsets = dict()

    def process_data_seq(self,
                         get_data_from_local_machine: bool,
                         dataset_name: str,
                         data_count: int,
                         subject_count_per_stage: int,
                         frequency: int,
                         sections: int,
                         channel_names: list,
                         channel_types: list,
                         process_step_name: str,
                         permitted_overwrite: bool,
                         do_trim_wakephases=False):     # @sp - add trimming of wakephases
        """
        Args:
            get_data_from_local_machine: Load raw data from a local folder instead.
            dataset_name: Name of the used dataset
            data_count: How many subjects should be used
            subject_count_per_stage: Todo: Janine add doc
            frequency: Sample frequency e.g. 200Hz
            sections: E.g. divided into 30s sections
            channel_names: name for each channel. Has a not None default
                inside the function.
            channel_types: Type for each channel. Has a not None default
                inside the function.
            process_step_name: The name for the actual working step
            permitted_overwrite: Allow to overwrite existent data.

        Description
        -----------
        Trimming, transforming ... dropping raw data.

        Params
        ------
        """
        file_path = get_raw_data_path(dataset_name, get_data_from_local_machine)

        if process_step_name not in self.offsets:
            self.offsets[process_step_name] = 0

        offset = self.offsets[process_step_name]

        self.experiment.load_seq(channel_names=channel_names,
                                 channel_types=channel_types,
                                 file_path=file_path, dataset_name=dataset_name,
                                 frequency=frequency, sections=sections,
                                 offset_seq=offset, num_subjects=data_count,
                                 do_trim_wakephases=do_trim_wakephases)     # @sp - add trimming of wakephases
        # self.offsets[process_step_name] += 1

    @staticmethod
    def __check_for_data(save_path: str, uuid: str):
        """
        Args:
            save_path: Path to the data
            uuid: uuid of the used experiment

        Description
        -----------
        Checks if the experiment exists.

        Params
        ------

        """
        if af.is_dir_in(path=save_path, dir_name=uuid) is not True:
            sys.exit("[ParameterError]: There's no data on this experiment. "
                     "Are you sure you want to load data for this experiment?")

    def load_particular_data(self, uuid: str, process_step_name: str,
                             num_subjects: int, params=None, offset: int = 0) -> np.array:     # @sp - add params

        """
        Args:
            uuid: Uuid of the used experiment
            process_step_name: Name of the actual process step
            data_count: Number of used subjects
            offset: Ignore the first x subjects

        :return: Data and classifications for the subjects

        Description
        ----
        Loads the already processed particular data from the subjects or files
            which have been saved by the user.

        Params
        ----

        """
        data_x, data_y = \
            self.experiment.load_stored_data(uuid=uuid,
                                             process_step_name=
                                             process_step_name,
                                             num_subjects=num_subjects,
                                             offset=offset,
                                             params=params)        # @sp - add params

        return data_x, data_y

    def store_particular_data(self, process_step_name: str, data_x: np.array,
                              data_y: np.array,
                              data_count: int, offset: int):
        """
        Args:
            process_step_name: Name of the process step to store the data
            data_x: The data to store as np array
            data_count: Number of used subjects
            offset: Ignore the first x subjects

        Description
        ----
        Saves the already processed particular data from the subjects or files
            which have been saved by the user.

        Params
        ----

        """
        self.experiment.store_data(process_step_name=process_step_name,
                                   data_x=data_x, data_y=data_y,
                                   data_count=data_count,
                                   offset=offset)

    @elapsed_time
    def get_next(self, process_step_name: str, uuid: str,
                 data_count: int, start_val: int = 0, params=None) -> np.array:     # @sp - add params
        """
        Args:
            start_val:
            process_step_name: Name of the used process step
            uuid: Uuid of the experiment
            data_count: Number of used subjects

        Description
        ----
        Get the data and classification from the next subject(s)

        Params
        ----

        """
        if process_step_name not in self.offsets:
            self.offsets[process_step_name] = start_val

        data_x = None
        data_y = None

        if self.offsets[process_step_name] < data_count:
            offset = self.offsets[process_step_name]

            data_x, data_y = self.load_particular_data(uuid=uuid,
                                                       process_step_name=
                                                       process_step_name,
                                                       num_subjects=1,
                                                       offset=offset)

        self.offsets[process_step_name] += 1
        if data_x is None or data_y is None:
            sys.exit("There is no data returned for the subject.")
        return data_x, data_y

    def get_distribution_matrix(self, count) -> object:
        subject_name = self.experiment.data_objects_list[count]
        save_path = self.experiment.save_path + subject_name
        loaded_object = load_pickle_to_obj(save_path=save_path, save_name="/distribution_matrix_preprocessed")
        return loaded_object

    def get_encoder(self) -> LabelEncoder:
        """
        :return: The LabelEncoder of the experiment

        Description
        ----
        Returns the LabelEncoder of the experiment

        """
        return self.experiment.encoder

    def get_list_processed(self, subject_range: tuple = None) -> list:
        """
        Args:
            subject_range: Range as tuple
        Returns: List of subjects

        Description
        ----
        Returns a list of x preprocessed subjects

        Params
        ----
        """
        if subject_range is None:
            data_object_list = self.experiment.data_objects_list
        else:
            data_object_list = self.experiment.data_objects_list[
                               subject_range[0]:subject_range[1]]
        return data_object_list
