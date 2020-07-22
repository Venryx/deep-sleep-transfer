import argparse  # polyaxon parameters
from typing import List


def get_parameters():
    """
    Gets the parser with its arguments and returns a dictionary of those args
    """
    parser = add_args()
    args = parser.parse_known_args()
    parameters = args[0].__dict__
    return parameters


def add_args():
    """Creates a parser object adds the arguments to it"""
    parser = argparse.ArgumentParser()
    _add_training_parameters(parser)
    _add_data_parameters(parser)
    _add_experimental_parameters(parser)
    _add_load_save_parameters(parser)
    return parser


def _add_training_parameters(parser):
    # -------------- Training ----------------------------------------------- #

    parser.add_argument(
        '--epochs',
        default=50,        # @sp - adjust training epochs
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=256,       # @sp - change batch size
        type=int
    )

# @sp - remove unused method

    parser.add_argument(
        # Set to -1 if no early stopping should be performed
        '--early_stopping_patience',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--one_epoch_per_fit',
        default=1,
        type=int
    )
    parser.add_argument(
        '--mdl_architecture',
        default="iitnet_cnn_bilstm",     # @sp - add IITNet, remove other models
        type=str
    )


def _add_load_save_parameters(parser):
    _add_load_parameters(parser)
    _add_save_parameters(parser)

    parser.add_argument(
        # Override normal save_path.
        '--save_path',
        # >>> @sp - add local and remote save paths
        default="D:/sleep-edf-v1/sleep-cassette/processed/training/",
        # "/resources/sa6pr7/sleep-edf-v1/sleep-cassette/processed/training/",    # Winslow sleep-edf
        # "/resources/sa6pr7/physionet_challenge/processed/training/",    # Winslow physionet
        # "D:/sleep-edf-v1/sleep-cassette/processed/training/",   # local sleep-edf
        # "D:/physionet_challenge/processed/sa6pr7/training/",  # local physionet
        # "D:/shhs1/processed/training/"    # local shhs
        # <<< @sp
        type=str
    )

    parser.add_argument(
        # filename under which the feature engineered data
        # is stored on the harddrive. Convention: <user>_<engKind>_<num>, e.g. st6kr5_morlet_1
        '--feature_eng_filename',
        default='sa6pr7' + '_raw_1',  # e.g. st6kr5_morlet_1    # @sp
        type=str
    )


def _add_load_parameters(parser):
    parser.add_argument(
        # decide to either load or generate data, encoder and uuid. Use this
        # to generate preprocessed files from the raw data.
        '--load',
        default=0,  # 1 if data is already preprocessed
        type=int
    )

    parser.add_argument(
        # decide which database to load.
        # currently supported: "preprocessed"(if you want generate from raw,
        # use this), "feature_eng"
        '--databasis',
        default="preprocessed",
        type=str
    )

    parser.add_argument(
        # a uuid if you want to load a specific experiment.
        # Don't use it with load=False, there is no reason
        # why you would do this.
        '--experiment_uuid',
        default="iitnet_0",     # @sp
        type=str
    )
    parser.add_argument(
        '--get_raw_data_from_local_path',
        # see confluence documentation for details on loading data
        default=1,  # @sp - 0=get data from server, 1=get local data
        type=int
    )
    parser.add_argument(
        '--dataset_name',
        # required if get_raw_data_from_local_path=True or if running on Marvin
        # implemented: deep_sleep and physionet_challenge
        default="deep_sleep",  # "deep_sleep",   # "physionet_challenge", # shhs1    # @sp - add SHHS dataset
        type=str
    )

    # >>> @sp - add possibility for already processed data
    parser.add_argument(
        '--data_already_processed',
        # if the data is already processed, load that for model training.
        default=True,
        type=bool
    )
    # <<< @sp


def _add_save_parameters(parser):
    parser.add_argument(
        '--save_raw_data',  # should the raw data also be saved?
        default=0,
        type=int
    )
    parser.add_argument(
        # decide if feature engineered data should be stored on the hard drive
        '--store_feature_eng',
        default=0,
        type=int
    )


def _add_data_parameters(parser):
    # ---------------- Data ------------------------------------------------- #

    parser.add_argument(
        # currently supported: "raw", "morlet_tsinalis", "ae"
        '--eng_kind',
        default="raw",
        type=str
    )
    parser.add_argument(
        '--subject_batch',  # Max number of subjects that should be loaded in at
        # a time
        default=1,  # @sp - for IITNet, this always is 1
        type=int
    )
    parser.add_argument(
        '--train_test_ratio',  # Set the train / test ratio. You need this
        # value and the train_count, then the test_value will get generated
        # automatically.
        default=None,
        type=int
    )
    parser.add_argument(
        '--train_count',  # how many subjects to use as training data.
        default=24,     # @sp - pretrain: 200; others: 24
        type=int
    )
    parser.add_argument(
        '--val_count',  # how many subjects to use as test data, not needed if a train_test_ratio is set
        default=6,     # @sp - pretrain: 50; others: 6
        type=int
    )

    parser.add_argument(
        '--test_count',  # how many subjects to use as test data, not needed if a train_test_ratio is set
        default=9,     # @sp - pretrain: 50; others: 9
        type=int
    )

    # @sp - remove unused method

    # refactor channel_types and channel_names and frequency to be hardcoded for each dataset and
    # be selected due to the choice of 'dataset_name'
    parser.add_argument(
        '--channel_types',  # type of each channel as list
        default=['eeg', 'eeg', 'eog', 'misc', 'emg', 'misc', 'misc'],    # sleep-edf    @sp - add sleep-edf
        # ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'misc',
        #         'eeg', 'eeg', 'misc', 'eeg'],  # physionet

        type=list
    )
    parser.add_argument(
        '--channel_names',  # name for each channel as list
        default=['Fpz-Cz', 'Pz-Oz', 'EOG', 'AIRFLOW', 'Chin1-Chin2', 'TEMP', 'EVENT'],  # sleep-edf  @sp - add sleep-edf
        # ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2',
        #         'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG'],  # physionet
        type=list
    )
    parser.add_argument(
        '--frequency',  # the sampling frequency
        # physionet_challenge data uses 200 Hz, deep_sleep uses 100 Hz, shhs: 125 Hz     @sp - add shhs
        default=100,
        type=int
    )
    parser.add_argument(
        # decide to either load or generate data, encoder and uuid. Use this
        # to generate preprocessed files from the raw data.
        '--ch_idx_list',
        default=[0],  # [3] for physionet, [0] for sleep-edf, [7] for shhs      @sp - add shhs
        type=List[int]
    )

    parser.add_argument(
        '--sections',  # the desired section e.g. 30 for 30 seconds
        default=30,
        type=int
    )
    parser.add_argument(
        '--ignore_warnings',
        default=1,
        type=int
    )
    parser.add_argument(
        '--permitted_overwrite',  # allow the program to overwrite
        # generated data. Handle with care! DON'T CHANGE IT.
        default=0,
        type=int
    )


def _add_experimental_parameters(parser):
    # ######################## Experimental variables #########################
    parser.add_argument(
        '--key_labels',  # not implemented yet. will help to select the relevant
        # classification classes
        default=["W", "N1", "N2", "N3", "R"],
        # default=['arousal', 'None'],

        # default=['arousal_rera', 'resp_centralapnea', 'resp_hypopnea',
        #                'resp_obstructiveapnea', 'None'],

        # default=['(arousal_rera', '(resp_centralapnea', '(resp_hypopnea',
        #         '(resp_obstructiveapnea', 'arousal_rera)',
        #         'resp_centralapnea)', 'resp_hypopnea)',
        #         'resp_obstructiveapnea)', 'arousal_rera', 'resp_centralapnea',
        #         'resp_hypopnea', 'resp_obstructiveapnea', 'None'],
        type=int
    )
    parser.add_argument(
        '--dummy',
        default=1,
        type=int
    )
