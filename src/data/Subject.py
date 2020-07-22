# pylint: disable=import-error, no-name-in-module

import os
import numpy as np
import mne
from util import auxiliary as af


class Subject:

    def __init__(self, subject_name: str = None,
                 subject_raw_units: np.array = None,
                 subject_raw_units_labels: dict = None,
                 subject_sample_frequency: int = None,
                 subject_sections: int = None,
                 subject_channel_names: list = None,
                 subject_channel_type: list = None):
        """
        Args:
            subject_name: Name of the subject e.g. patient
            subject_raw_units: A list of single units e.g. nights
            subject_raw_units_labels: A list of classifications e.g. w, n1,
            subject_sample_frequency: Sample frequency e.g. 200Hz
            subject_sections: E.g. data will be divided into 30s sections
            subject_channel_names: Name for each channel. Has a not None default
                inside the function.
            subject_channel_type: Type for each channel. Has a not None default
                inside the function.
        Description
        ----
        This is an abstraction for our patients. it should make it possible
        to manage and use the data more easily.

        Params
        ----

        """
        # -- raw --
        self.subject_uuid = af.create_uuid(True)
        self.subject_name = subject_name
        self.subject_raw_units = list()
        self.subject_raw_units.append(subject_raw_units)
        self.subject_raw_units_labels = list()
        self.subject_raw_units_labels.append(subject_raw_units_labels)
        self.subject_sample_frequency = subject_sample_frequency
        self.subject_sections = subject_sections
        self.subject_channel_names = subject_channel_names
        self.subject_channel_type = subject_channel_type
        # -- processed --
        self.subject_processed_units = list()
        self.subject_processed_units_labels = list()
        self.subject_indices = []
        self.subject_stored_data = dict()
        self.subject_stored_data_list = list()
        self.subject_stored_labels = dict()

    def __add__(self, other):
        self.subject_raw_units = \
            np.append(self.subject_raw_units, other.subject_raw_units, axis=0)
        self.subject_raw_units_labels = \
            np.append(self.subject_raw_units_labels,
                      other.subject_raw_units_labels, axis=0)
        # self.subject_raw_units.append(other.subject_raw_units)
        # self.subject_raw_units_labels.append(other.subject_raw_units_labels)
        self.subject_processed_units = \
            np.append(self.subject_processed_units,
                      other.subject_processed_units, axis=0)
        self.subject_processed_units_labels = \
            np.append(self.subject_processed_units_labels,
                      other.subject_processed_units_labels, axis=0)

    def __eq__(self, other):
        if self.subject_name == other.subject_name and \
                self.subject_channel_names == other.subject_channel_names and \
                self.subject_channel_type == other.subject_channel_type and \
                self.subject_sections == other.subject_sections and \
                self.subject_sample_frequency == other.subject_sample_frequency:
            return True
        return False

    def __repr__(self):
        return f"Subject: {self.subject_uuid}\n" \
               f"Name: {self.subject_name}\n" \
               f"Frequency: {self.subject_sample_frequency}\n" \
               f"Samples: {len(self.subject_processed_units)}"

    def get_size(self) -> int:
        """
        :return: Shape

        Description
        ----
        Return the size of the raw data

        Params
        ----
        """
        return len(self.subject_raw_units)

    def create_indices(self):
        """

        Return the size of the raw data
        """
        for element in self.subject_processed_units:
            self.subject_indices.append(element.shape[0])

    def get_sum_indices(self) -> int:
        """
        :return: Indices means the number of sections for this subject
            e.g. patient

        Description
        ----
        Get the number of sections for this subject e.g. patient

        Params
        ----

        """
        indices_sum = 0
        for element in self.subject_indices:
            indices_sum += element
        return indices_sum

    def get_channel_count(self) -> int:
        """
        :return: int

        Description
        ----
        Return count of input_channels.

        Params
        ----
        """
        return len(self.subject_channel_names)

    def append_raw_units(self, raw_unit: np.array):
        """
        Args:
            raw_unit: A new unit e.g. night

        Description
        ----
        Append the list of raw units e.g. nights with one entry(raw_unit).

        Params
        ----

        """
        # Todo: is raw_unit a np.array
        self.subject_raw_units.append(raw_unit)

    def append_processed_units(self, processed_unit: np.array):
        """
        Args:
            processed_unit: A new unit e.g. night (processed)


        Description
        ----
        Append the list of processed units e.g. nights with one entry.

        Params
        ----

        """
        self.subject_processed_units.append(processed_unit)

    def plot_mne(self,
                 mode: str = 'raw',
                 scaling: dict = None,
                 unit_id: int = 0,
                 position: int = 0) -> object:
        """
        Args:
            mode:  which data should be plotted e.g. raw or processed
            scaling: how to scale the different types of channels
                e.g. {'eeg': 20}
            unit_id: the unit of the subject e.g. night
            position: Ignore the first X seconds and start at second

        Description
        ----
        Allows to plot the time series of the object.

        Hints
        ----
        Currently only raw data is supported

        Params
        ----

        """

        if scaling is None:
            scaling = {'eeg': 20, 'misc': 200}

        offset = position * self.subject_sample_frequency
        info = mne.create_info(ch_names=self.subject_channel_names,
                               sfreq=self.subject_sample_frequency,
                               ch_types=self.subject_channel_type)

        if mode == 'raw':
            try:
                plot_data = self.subject_raw_units[unit_id][:, offset:]

                plot_type = 'Raw'
            except:
                print("No raw data for this object.")
                return 0
        # TODO:
        #       add function to plot individual data
        #       elif ... mode e.g. process_step_name
        else:
            unit_data = self.subject_processed_units[unit_id]
            placeholder_plot_data = self.generate_placeholder(unit_data)
            tmp_data = np.transpose(unit_data, (2, 1, 0))
            index_counter = 0
            for _ in tmp_data:
                placeholder_plot_data[index_counter] = tmp_data[
                    index_counter].flatten(order='A')
                index_counter += 1
            plot_data = placeholder_plot_data
            plot_type = 'Processed'

        data = mne.io.RawArray(plot_data, info)
        data.plot(n_channels=self.get_channel_count(), scalings=scaling,
                  title=(f"{plot_type} data from subject \'"
                         f"{self.subject_name}\' ### unit: {str(unit_id)}"
                         f" ### offset: {str(position)} seconds ###"),
                  show=True, block=True)

    def generate_placeholder(self, unit_data: np.array) -> np.array:
        """
        Args:
            unit_data:

        Returns: A zero padded numpy array with shape of the given data

        """
        data_points_per_second = \
            self.subject_sample_frequency * self.subject_sections

        act_data_points = int(unit_data.shape[0] * data_points_per_second)
        shape = (self.get_channel_count(), act_data_points)
        return np.zeros(shape, )

    def store_data(self, process_step_name: str, data_x: np.array,
                   data_y: np.array,
                   save_path: str) -> None:
        """
        Args:
            process_step_name: The name under which the stored data should
            be saved and reloaded later.
            data_x: Data as array, which should be stored
            save_path: The path in which the data is to be stored.

        Description
        ----
        Stores arbitrary data under a specific name inside this subject.

        Params
        ----

        """
        unit_dict = dict()
        label_dict = dict()
        indices_offset = 0

        for i, element in enumerate(self.subject_indices):
            unit_dict[i] = data_x[indices_offset:indices_offset + element]
            label_dict[i] = data_y[indices_offset:indices_offset + element]
            np.save(f"{save_path}{self.subject_name}/{str(i)}"
                    f"_data_x_{process_step_name}.npy", unit_dict.get(i))

            np.save(f"{save_path}{self.subject_name}/{str(i)}"
                    f"_data_y_{process_step_name}.npy", label_dict.get(i))

            indices_offset += element

        self.subject_stored_data[process_step_name] = unit_dict
        self.subject_stored_labels[process_step_name] = label_dict
        self.subject_stored_data_list.append(process_step_name)

    def load_stored_data(self, process_step_name: str, save_path: str) -> None:
        """
        Args:
            process_step_name: The name under which the stored data should
            be saved and reloaded later.
            save_path: The path in which the data is to be stored

        Description
        ----
        Load arbitrary data under a specific name from this subject.

        Params
        ----

        """
        files = os.listdir(save_path + self.subject_name)
        count_data_x_data = str(
            os.listdir(save_path + self.subject_name)).count("_data_x_" +
                                                             process_step_name +
                                                             ".npy")
        tmp_x_array = dict()
        for i in range(0, count_data_x_data):
            tmp_x_array[i] = (np.load(f"{save_path}{self.subject_name}/"
                                      f"{str(i)}_data_x_{process_step_name}.npy"))

        count_data_y_data = str(
            os.listdir(save_path + self.subject_name)).count("_data_y_" +
                                                             process_step_name +
                                                             ".npy")
        tmp_y_array = dict()
        for i in range(0, count_data_y_data):
            tmp_y_array[i] = (np.load(f"{save_path}{self.subject_name}/"
                                      f"{str(i)}_data_y_{process_step_name}.npy"))

        self.subject_stored_data[process_step_name] = tmp_x_array
        self.subject_stored_labels[process_step_name] = tmp_y_array
        print()

    def get_stored_data(self, process_step_name: str) -> np.array:
        """
        :param process_step_name: The name under which the stored data should
            be saved and reloaded later. The name under which the stored data
            should be saved and reloaded later.
        :return:  the loaded arbitrary data for this subject

        Description
        ----
        Load arbitrary data under a specific name from this subject.

        Params
        ----

        """
        data_x = None
        for element in self.subject_stored_data.get(process_step_name):
            if data_x is None:
                data_x = self.subject_stored_data.get(process_step_name).get(
                    element)
            else:
                data_x = np.append(data_x, self.subject_stored_data.get(
                    process_step_name).get(element), axis=0)
        return data_x

    def get_stored_labels(self, process_step_name):
        """Missing TODO"""
        data_y = None
        for element in self.subject_stored_labels.get(process_step_name):
            if data_y is None:
                data_y = self.subject_stored_labels.get(process_step_name).get(
                    element)
            else:
                data_y = np.append(data_y,
                          self.subject_stored_labels.get(process_step_name).get(
                              element), axis=0)
        return data_y

    # def get_labels(self):
    #     """Missing TODO"""
    #     data_y = None
    #     for element in :
    #         if data_y is None:
    #             self.subject_stored_labels.get(process_step_name).get(
    #                 element)
    #         else:
    #             np.append(data_y, self.subject_stored_labels.get(process_step_name).get(
    #                 element), axis=0)
    #     return data_y

    def dump(self, save_path: str, save_raw: bool = True) -> object:
        """
        :param save_path: path where the object should be stored.
        :param save_raw: should the raw data be saved

        Description
        ----
        Dump an object e.g. patient to a file and subfiles

        Params
        ----

        """
        af.check_and_make_path(save_path + self.subject_name)

        if save_raw:
            if self.subject_raw_units is not None:
                for index, element in enumerate(self.subject_raw_units):
                    np.save(save_path + self.subject_name + "/" + str(
                        index) + "_data_x_raw.npy", element)

                for index, element in enumerate(self.subject_raw_units_labels):
                    np.save(save_path + self.subject_name + "/" + str(
                        index) + "_data_y_raw.npy", element)

        for index, element in enumerate(self.subject_processed_units):
            np.save(save_path + self.subject_name + "/" + str(
                index) + "_data_x_preprocessed.npy", element)

        for index, element in enumerate(self.subject_processed_units_labels):
            np.save(save_path + self.subject_name + "/" + str(
                index) + "_data_y_preprocessed.npy", element)


        self.subject_processed_units = list()
        self.subject_processed_units_labels = list()
        self.subject_raw_units = None
        self.subject_raw_units_labels = None
        af.name_subject_dump(self, self.subject_name,
                             save_path + self.subject_name + "/")

    def load_from_file(self, save_name: str, save_path: str) -> object:
        """
        :param save_name: name of the file
        :param save_path: path where the object is stored.

        Description
        ----
        Load an object e.g. patient form a file.

        Params
        ----

        """
        save_path_tmp = save_path + save_name + "/"
        loaded_object = af.load_pickle_to_obj(save_name=save_name,
                                              save_path=save_path_tmp)
        self.subject_uuid = loaded_object.subject_uuid
        self.subject_name = loaded_object.subject_name
        self.subject_sample_frequency = loaded_object.subject_sample_frequency
        self.subject_sections = loaded_object.subject_sections
        self.subject_channel_names = loaded_object.subject_channel_names
        self.subject_channel_type = loaded_object.subject_channel_type
        self.subject_indices = loaded_object.subject_indices
        # self.subject_stored_data_list = loaded_object.subject_stored_data_list
        self.subject_processed_units = list()
        self.subject_processed_units_labels = list()
        self.subject_raw_units = list()
        self.subject_raw_units_labels = list()

        count_train_x_preprocessed = str(
            os.listdir(save_path + save_name)).count(
            "_data_x_preprocessed.npy")
        for i in range(0, count_train_x_preprocessed):
            self.subject_processed_units.append(
                np.load(save_path_tmp + str(i) + "_data_x_preprocessed.npy"))

        count_train_y_preprocessed = str(
            os.listdir(save_path + save_name)).count(
            "_data_y_preprocessed.npy")
        for i in range(0, count_train_y_preprocessed):
            self.subject_processed_units_labels.append(
                np.load(save_path_tmp + str(i) + "_data_y_preprocessed.npy"))

        count_train_x_raw = str(os.listdir(save_path + save_name)).count(
            "_data_x_raw.npy")
        for i in range(0, count_train_x_raw):
            self.subject_raw_units.append(
                np.load(save_path_tmp + str(i) + "_data_x_raw.npy"))
