# pylint: disable=import-error, no-name-in-module
import os
import sys

import numpy as np
import scipy.io
import wfdb

from data import Subject


def __load_file(file_path: str, file_name: str) -> []:
    """
    Args:
        file_path: E.g path to the folder on the server
        file_name: Name of the file

    Returns: Array with raw time series data.

    Description
    ----
    Loading raw data to work with from a file. (e.g. a single night)
    Without the corresponding annotations.

    Params
    ----
    """
    file_input = scipy.io.loadmat(f"{file_path}{file_name}/{file_name}")
    return file_input['val']


def __load_annotations(path: str, column_name: str) -> dict:
    """
    Args:
        path: path to the annotations file (e.g. "//srv39.itd-intern.de/
        MachineLearningDatasets/")
        column_name:  name of the column in array (e.g. "arousal")
    Returns: dict of annotations with position (e.g. [96000: 'W'])

    Description
    ----
    Loading annotations corresponding to the raw data. (e.g. for a single night)
    Without the corresponding raw data.

    Params
    ----
    """
    annotations = wfdb.rdann(path, column_name)
    clean_annotations = np.asarray(annotations.aux_note)
    sample_annotations = annotations.sample
    annotations_dict = dict()
    for i, element in enumerate(sample_annotations):
        annotations_dict[element] = clean_annotations[i]

    return annotations_dict


def load_data(file_path=f"//srv11.itd-intern.de/MachineLearningDatasets"
                        f"/medical/physionet_challenge/training/",
              frequency=200,
              sections=30,
              channel_names=None,
              channel_types=None,
              offset=0):
    """
    Args:
        file_path: E.g path to the folder on the server
        frequency: Sampling frequency
        sections: E.g. divided into 30s sections
        channel_names: name for each channel. Has a not None default
        inside the function.
        channel_types: Type for each channel. Has a not None default
        inside the function.
        offset: Ignore the first x files.
    Returns: A Subject object

    Description
    ----
    Loading transform raw data and the corresponding annotations into a
    >>Subject<<. This simplifies the handling of different sources.

    Params
    ----

    """
    if channel_names is None:
        channel_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
                         'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST',
                         'AIRFLOW', 'SaO2', 'ECG']

    if channel_types is None:
        channel_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                         'misc', 'eeg', 'eeg', 'misc', 'eeg']

    try:
        file_list = os.listdir(file_path)
    except FileNotFoundError:
        sys.exit("\n[ParameterError]: Can not find files on the server. "
                 "Are you sure you connected to the filesystem?")

    file_list.sort()
    # sort the file_list to ensure that the data is in the same order on
    # every system

    try:
        file = str(file_list[offset])
    except IndexError:
        sys.exit("\n[IndexError]: There is no file left to load.")

    subject = Subject.Subject(subject_name=file,
                              subject_raw_units=__load_file(file_path, file),
                              subject_raw_units_labels=
                              __load_annotations(f"{file_path}{file}/{file}",
                                                 "arousal"),
                              subject_sample_frequency=frequency,
                              subject_sections=sections,
                              subject_channel_names=channel_names,
                              subject_channel_type=channel_types)

    return subject
