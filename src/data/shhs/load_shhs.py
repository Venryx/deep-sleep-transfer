# @sp: newly created

import os
import sys

import mne
import xmltodict

from data import Subject


def __load_file(file_path_psg,
                files_psg,
                offset=0):
    data = mne.io.read_raw_edf(file_path_psg + files_psg[offset])
    raw_data = data.get_data()

    channel_names = data.ch_names

    return raw_data, channel_names


def __load_labels(file_path_hyp,
                  files_hyp,
                  offset=0):
    annotation_event_id = {'Wake|0': 'W',
                           'Stage 1 sleep|1': 'N1',
                           'Stage 2 sleep|2': 'N2',
                           'Stage 3 sleep|3': 'N3',
                           'Stage 4 sleep|4': 'N3',  # not used
                           'REM sleep|5': 'R',
                           'Sleep stage ?': 'Sleep stage ?',
                           'Movement time': 'Movement time'}

    sampling_rate = 125     # TODO: Read from params?

    f_path = file_path_hyp + files_hyp[offset]
    with open(f_path) as fd:
        xml = xmltodict.parse(fd.read())

    all_annotations = xml['PSGAnnotation']
    all_events = all_annotations['ScoredEvents']
    all_events_list = all_events['ScoredEvent']
    sleep_events_list = []
    for element in all_events_list:
        if element['EventType'] == 'Stages|Stages':
            sleep_events_list.append(element)

    annotations_dict = dict()
    for element in sleep_events_list:
        start = float(element['Start'])
        start = start * sampling_rate  # get the start in datapoints, not seconds
        label = element['EventConcept']
        converted_label = annotation_event_id[label]

        annotations_dict[int(start)] = converted_label

    return annotations_dict


def load_data(file_path,  # "Z:/shhs/polysomnography/"
              frequency=125,
              sections=30,
              channel_names=None,
              channel_types=None,
              offset=0):
    channel_types = ['misc', 'misc', 'eeg', 'ecg', 'emg', 'eog', 'eog',
                     'eeg', 'misc', 'misc', 'misc', 'misc', 'misc', 'misc']

    file_path_hyp = file_path + "/annotations-events-nsrr/shhs1/"
    file_path_psg = file_path + "/edfs/shhs1/"

    try:
        files_hyp = [file for file in os.listdir(file_path_hyp)]
        files_psg = [file for file in os.listdir(file_path_psg)]
    except FileNotFoundError:
        sys.exit("\n[ParameterError]: Can not find files on the server. "
                 "Are you sure you connected to the filesystem?")

    files_hyp.sort()
    files_psg.sort()

    subject_name = files_psg[offset].strip('.edf')

    data, channel_names = __load_file(file_path_psg=file_path_psg,
                                      files_psg=files_psg,
                                      offset=offset)

    raw_units_labels = __load_labels(file_path_hyp=file_path_hyp,
                                     files_hyp=files_hyp,
                                     offset=offset)

    subject = Subject.Subject(subject_name=subject_name,
                              subject_raw_units=data,
                              subject_raw_units_labels=raw_units_labels,
                              subject_sample_frequency=frequency,
                              subject_sections=sections,
                              subject_channel_names=channel_names,
                              subject_channel_type=channel_types)

    return subject
