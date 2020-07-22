# @sp: newly created

import datetime
import os
import sys

from keras.utils.layer_utils import count_params

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')
sys.path.append('/resources/')

import params.polyaxon_parsing_iitnet_cnn_lstm as pp3

from keras.models import load_model
from data.process_data import process_data
from models.model_functions import choose_model
from params.Params_Winslow import Params, winslow_params
from models.architectures.iitnet_cnn_bilstm import compile_model_iitnet
from data.DataInterface import DataInterface as DataInt
from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator
from trainings.paramtuning_old.aux_tuning import record_performance
from keras import backend as K


def train_iitnet_crossvalid():
    """
    Train the iitnet using cross validation. Using only training and validation data
    for parameter tuning. Best model will then be evaluated in a separate program.
    """

    # log timestamps of relevant stages
    start_processing = datetime.datetime.now()
    timestamps = {'processstart': start_processing}

    # print used devices
    print("Using GPU:", K.tensorflow_backend._get_available_gpus())

    # get parameters
    params = Params()
    # get additional parameters for iitnet
    plx: dict = pp3.get_parameters()
    params.plx.update(plx)

    # adjust winslow parameters
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        winslow_params(params)

    params.plx['subject_batch'] = 1  # !
    # NOTE: mdl_architecture has to be set to 'iitnet_cnn_bilstm'

    # set local parameters for the cross validation
    k = params.plx.get('k_crossval')
    train_total = params.plx.get('train_count') + params.plx.get('val_count')
    count_per_fold = train_total // k

    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    # Process data, if not already processed
    if not params.plx.get("data_already_processed"):
        # Process Data
        process_data(params, data_int, params.plx["data_count"])
    else:
        # recover self.experiment.data_objects_list = List of the subject names
        preprocessed_data_path = params.plx["save_path"] + params.plx["experiment_uuid"]    # "D:/PhysioNet/processed/sa6pr7/"
        pickle_object = params.plx["experiment_uuid"] + ".pckl"
        subject_folders = [name for name in os.listdir(preprocessed_data_path) if not name == pickle_object]

        relevant_subjects = subject_folders[:train_total]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)

    num_epochs = params.plx.get('epochs')
    apply_oversampling = params.plx.get('apply_oversampling')  # !only on training data

    timestamps['modelstart'] = datetime.datetime.now()
    # build model
    model, callbacks = choose_model(params, compile=False)
    timestamps['modelend'] = datetime.datetime.now()
    # save untrained model
    if k > 1:
        print("Save untrained model ... ", end=" ")
        model.save(params.file_path_raw_mdl)
        print("done")

    timestamps_trainingstart = []
    timestamps_trainingend = []
    all_val_accs = []
    all_val_loss = []
    timestamps['crossval_start'] = datetime.datetime.now()

    for i in range(k):
        print("\n=============================================")
        print("=======> Cross Validation - Fold #", i + 1, "<=======")
        print("=============================================")

        # get raw model
        if k > 1:
            print("Load untrained model ... ", end=" ")
            model = load_model(params.file_path_raw_mdl)
            print("done")
        # compile model
        model = compile_model_iitnet(params=params, model=model)

        # set indices for the data to be loaded in this fold
        if k == 1:
            train_start = 0
            train_end = int(train_total * 0.8)
            val_start = train_end
            train_count = train_end
            val_count = train_total - train_count
        else:
            train_start = i * count_per_fold
            train_end = train_start + (count_per_fold * (k - 1))
            if train_end >= train_total:
                train_end -= train_total
            val_start = train_end
            if val_start >= train_total:
                val_start = 0

            # configure the data generators for training and validation
            train_count = train_total - count_per_fold
            val_count = count_per_fold

        train_generator = InterIntraEpochGenerator(data_int, params, train_count, start_val=train_start, shuffle=True,
                                                   oversampling=apply_oversampling, crossval_samples=train_total)
        validation_generator = InterIntraEpochGenerator(data_int, params, val_count,
                                                        start_val=val_start, crossval_samples=train_total)

        # model training
        print("####\n\n\nTraining###\n\n")
        timestamps_trainingstart.append(datetime.datetime.now())

        history = model.fit_generator(generator=train_generator,
                                      epochs=num_epochs,
                                      callbacks=callbacks,
                                      workers=0,
                                      validation_data=validation_generator,
                                      use_multiprocessing=False)

        timestamps_trainingend.append(datetime.datetime.now())
        print("Model Training done. Save Performance to Log ... ", end=" ")

        # log the performance of this fold
        val_acc_history = history.history['val_accuracy']        # val_accuracy for Winslow, val_acc local
        val_loss_history = history.history['val_loss']

        all_val_accs.append(val_acc_history)
        all_val_loss.append(val_loss_history)
        print("done.")

    print("=======> Cross Validation - Performance Evaluation <=======")
    timestamps['crossval_end'] = datetime.datetime.now()
    timestamps['trainstarts'] = timestamps_trainingstart
    timestamps['trainends'] = timestamps_trainingend
    train_parameters = count_params(model.trainable_weights)
    record_performance(all_val_accs, all_val_loss, params, timestamps, train_parameters)


if __name__ == "__main__":
    train_iitnet_crossvalid()
