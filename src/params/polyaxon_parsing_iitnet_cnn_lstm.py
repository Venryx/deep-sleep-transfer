# @sp: newly created

import argparse  # polyaxon parameters


def get_parameters():
    """
    Additional parameters for the IITNet model
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        # number of epochs to be considered for predicting one sleep phase
        '--l_epochs',
        default=3,     #paper: 11
        type=int
    )

    parser.add_argument(
        # number of parts that one epoch is to be split into
        '--l_subepochs',
        default=3,
        type=int
    )

    parser.add_argument(
        # how much the subepochs overlap (in percent)
        '--l_overlapping',
        default=10,
        type=int
    )

    parser.add_argument(
        # number of filters in the first stage of iitnet
        '--filtersize',
        default=8,
        type=int
    )

    parser.add_argument(
        # if the samples should be oversampled
        '--apply_oversampling',
        default=False,
        type=bool
    )

    parser.add_argument(
        # if the data should be downsampled to a certain frequency
        '--apply_downsampling',
        default=True,
        type=bool
    )

    parser.add_argument(
        # frequency to downsample the signal data to
        '--common_frequency',
        default=100,
        type=int
    )

    parser.add_argument(
        # whether to trim long wakephases at the end and beginning
        '--do_trim_wakephases',
        default=True,
        type=bool
    )

    parser.add_argument(
        '--load_model',
        default=False,
        type=bool
    )

    parser.add_argument(
        # k for k-fold cross validation. -1 means no cross validation
        # currently not used (see paramtuning_old)
        '--k_crossval',
        default=-1,
        type=int
    )

    parser.add_argument(
        # learning rate for the model optimizer
        '--lr',
        default=0.00001,    # 0.0005 for pretrain, 0.00001 for finetuning
        type=float
    )

    parser.add_argument(
        # Beta 1 for the model optimizer, as defined in the paper
        '--beta1',
        default=0.9,
        type=float
    )

    parser.add_argument(
        # Beta 2 for the model optimizer, as defined in the paper
        '--beta2',
        default=0.999,
        type=float
    )

    parser.add_argument(
        '--epsilon',
        default=1.0e-8,
        type=float
    )

    parser.add_argument(
        '--l2_regularization_factor',
        default=1.0e-6,
        type=float
    )

    parser.add_argument(
        '--tl_mode',
        default='direct',  # "pretrain", 'pretrain_multiple', "scratch", "direct", "dense", "recurrent", "cnn", "complete",
        type=str
    )

    parser.add_argument(
        '--tl_stage',
        default="evaluation",     # "training", "evaluation",
        type=str
    )

    args = parser.parse_known_args()
    parameters = args[0].__dict__
    return parameters
