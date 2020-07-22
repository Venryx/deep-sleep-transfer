import os
import sys

import mne

from data import Subject


def load_annotations(filepath):
    annotations = mne.read_annotations(filepath)

    annotation_event_id = {'Sleep stage W': 'W',
                           'Sleep stage 1': 'N1',
                           'Sleep stage 2': 'N2',
                           'Sleep stage 3': 'N3',
                           'Sleep stage 4': 'N3',
                           'Sleep stage R': 'R',
                           'Sleep stage ?': 'Sleep stage ?',
                           'Movement time': 'Movement time'}

    annotations_dict = dict()
    last = 0  # to remember the index of the sample of the last label
    for index, desc in enumerate(annotations.description):
        duration = int(
            annotations.duration[index] / 30)  # number of 30s epochs
        # for a label
        annotations_dict[(duration + last) * 3000] = annotation_event_id[desc]
        # duration * 3000 -> number of samples for the label
        last += duration
    return annotations_dict


def load_data(file_path,
              frequency=100,
              sections=30,
              channel_names=None,
              channel_types = None,
              offset=0):

    if channel_types is None:
        channel_types = ['eeg', 'eeg', 'eog', 'misc', 'emg', 'misc',
                             'misc']

    try:
        files_hyp = [file for file in os.listdir(file_path) if
                     file.endswith('Hypnogram.edf')]
        files_psg = [file for file in os.listdir(file_path) if
                     file.endswith('PSG.edf')]
    except FileNotFoundError:
        sys.exit("\n[ParameterError]: Can not find files on the server. "
                 "Are you sure you connected to the filesystem?")

    files_psg.sort()

    subject_name = files_psg[offset].strip('-PSG.edf')

    data = mne.io.read_raw_edf(file_path + files_psg[offset])
    raw_data = data.get_data()
    channel_names = data.ch_names

    data = None
    files_psg = None

    files_hyp.sort()
    raw_units_labels = load_annotations(file_path + files_hyp[offset])
    files_hyp = None

    subject = Subject.Subject(subject_name=subject_name,
                              subject_raw_units=raw_data,
                              subject_raw_units_labels=raw_units_labels,
                              subject_sample_frequency=frequency,
                              subject_sections=sections,
                              subject_channel_names=channel_names,
                              subject_channel_type=channel_types)

    return subject
