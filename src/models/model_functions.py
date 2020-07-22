import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import get_custom_objects
import tensorflow as tf

# @sp - remove unnecessary imports
from models.architectures.iitnet_cnn_bilstm_shared import create_model_iitnet   # @sp - add IITNet
from util import model_evaluation_utils as meu
from util.input_path_utils import get_src_parent_dir


# @sp - remove unused method


def define_callbacks_for_fit(params):
    """
    :param params:
    :return:

    """
    callbacks_list = [TensorBoard(log_dir=params.logdir_tb),
                      ModelCheckpoint(params.file_path_mdl, monitor='val_loss',
                                      mode='min',
                                      verbose=1, save_best_only=True,
                                      save_weights_only=True)]
    if params.plx.get('early_stopping_patience') != -1:
        callbacks_list.append(
            EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                          patience=params.plx.get(
                              'early_stopping_patience'),
                          restore_best_weights=True))
    return callbacks_list


def choose_model(params, model=None, do_compile=True, no_model=False):  # @sp - add parameters do_compile and no_model
    mdl_architecture = params.plx.get("mdl_architecture")

    callbacks_list = define_callbacks_for_fit(params)

    if model is None:
        # >>> @sp - add IITNet
        if mdl_architecture == 'iitnet_cnn_bilstm':
            if params.plx.get("load_model") or no_model is True:
                model = None
            else:
                model = create_model_iitnet(params, model, do_compile)
            return model, callbacks_list
        # <<< @sp
        else:
            # >>> @sp - remove all other models except IITNet
            raise ValueError(
                f"Value \"{mdl_architecture}\" for parameter \"mdl_architecture\" "
                f"is not supported")
            # <<< @sp

    return model, callbacks_list


# @sp - remove unused method


def save_model(model, file_name):
    mdl_path = get_src_parent_dir() + "src/trainings/mdl_chkpts/" + file_name
    model.save(mdl_path)


# @sp - remove unused method


def get_num_steps(params):
    eng_kind = params.plx.get('eng_kind')
    if eng_kind == 'morlet_tsinalis':
        num_steps = 314
    elif eng_kind == 'ae':
        num_steps = 128
    else:
        num_steps = params.plx.get('sections') * params.plx.get('frequency')
    num_steps = num_steps * (2 * params.temporal_context + 1)
    return num_steps


# @sp - remove unused method


def get_neighbours(t_x_red, t_y_red, temporal_context, start, end,
                   n_channels):
    """
    Takes an array of data and labels and splits them into batches depending on
    the temporal context.

    Examples:

    If temporal_context=0 then this will just return
    t_x = t_x_red[start:end]

    For temporal_context=2: for each window we also take the samples from the 2
    prior windows and the 2 next windows. The number of samples for entry of t_x
    will contain 5*original_num_samples, since we are considering samples from
    5 windows altogether.

    :param temporal_context: how many windows either side should be considered
    in each batch taken.
    :param t_x_red: np.array of shape (num_sections, num_channels, num_samples)
    containing the data to be split into batches.
    :param t_y_red: np.array of shape (num_sections, l), where l is the label
    for the section.
    :param start: Which window should be the first centre window.
    N.B should be that start>=temporal context. If not, for example start=0,
    temporal_context=1 then our first batch would contain data from
    window '-1', window 0 and window 1.
    :param end: The centre window for the last batch
    :param n_channels: num_channels for data
    :return: t_x: np.array of shape (batch_size, num_channels, new_num_samples)
    where the batch size = end-start, and
    new_num_samples=original_num_samples*((2 * temporal_context) + 1)
    t_y: np.array of shape (batch_size, l) where l is the label for each batch
    """
    t_x = []
    n_channels = t_x_red.shape[1]
    n_samples = t_x_red.shape[2] * ((2 * temporal_context) + 1)
    for time_period in range(start, end):
        t_x_new = []
        for channel in range(n_channels):
            for samples in range(-temporal_context, temporal_context + 1):
                t_x_new.append(t_x_red[time_period + samples, channel])
        t_x.append(t_x_new)
    t_x = np.array(t_x)
    t_x = np.reshape(t_x, (-1, n_channels, n_samples))

    t_y = np.copy(t_y_red[start:end])
    print(f"batch from index {start} to {end}")
    return t_x, t_y


def call_get_neighbours(batch, batch_size, data_x, data_y, n_channels,
                        temporal_context):
    if data_x.shape[0] >= (batch + 1) * batch_size:
        end = batch * batch_size + batch_size
        if batch == 0:
            start = temporal_context
        else:
            start = batch * batch_size
    else:
        end = data_x.shape[0] - temporal_context
        start = batch * batch_size

    t_x, t_y = get_neighbours(data_x, data_y,
                              temporal_context,
                              start,
                              end,
                              n_channels)
    return t_x, t_y


def set_batch_start_end(batch, batch_size, data_x, temporal_context):
    if batch == 0:
        start = temporal_context
    else:
        start = batch * batch_size

    if data_x.shape[0] >= (batch + 1) * batch_size:
        end = batch * batch_size + batch_size
    else:
        end = data_x.shape[0] - temporal_context
    return start, end


def initial_epoch_generator(params, num_epochs):
    """

    :param params:
    :param num_epochs:
    :return epoch_offset: used in fit_generator (=initial_epoch)
    """
    epoch_offset = 0
    while True:
        yield epoch_offset
        epoch_offset += num_epochs


def calc_steps(count, params, distribution_matrix, start_val=0):
    """
     :param count: number of subjects to be used in generator
     :param params: polyaxon parameter
     :param distribution_matrix: contains information about the
     class-distribution of the used subjects
     :param start_val: index of the first subject

     :return steps:

    Description
    -----------
    calcs steps for fit_generator: steps defines how often fit_generator has to
    call the generator/how many batches the generator yields

    """
    steps = 0
    batch_size = params.plx.get('batch_size')
    num_labels = 0
    n_subjects = params.plx.get('subject_batch')
    # >>> @sp - account for IITNet
    model_architecture = params.plx.get('mdl_architecture')
    l_epochs = params.plx.get('l_epochs')
    # <<< @sp

    for object_count in range(0, count, n_subjects):
        # i.e. if last batch, take whatever is left
        if count - object_count < n_subjects:
            n_subjects = count - object_count
        # object count is like a batch of subjects
        for subject_count in range(n_subjects):
            labels_subject = sum(distribution_matrix[
                                     object_count + start_val + subject_count].values())

            # >>> @sp - account for IITNet
            if model_architecture == "iitnet_cnn_bilstm":
                labels_subject = labels_subject - (l_epochs - 1)

            num_labels += labels_subject
            # <<< @sp

        steps += np.ceil(num_labels / batch_size)
        num_labels = 0

    return int(steps)


def calc_class_weights():
    pass

# @sp - remove unused methods
