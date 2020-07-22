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
from evaluation.evaluation_obj import Eval
from params.Params_Winslow import Params, winslow_params
from data.DataInterface import DataInterface as DataInt
from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator
from trainings.paramtuning_old.aux_tuning import record_performance
from keras import backend as K


def train_iitnet_paramtuning():
    """
    Hyperparametertuning for the IITNet model. Only using validation and
    training data to avoid information leakage to test data.
    """

    # log timestamps of relevant stages
    start_processing = datetime.datetime.now()
    timestamps = {'processstart': start_processing}

    # print used devices
    print("Using GPU:", K.tensorflow_backend._get_available_gpus())

    # PARAMETERS
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

    # DATA
    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    # Process data, if not already processed
    train_total = params.plx.get('train_count') + params.plx.get('val_count')
    if not params.plx.get("data_already_processed"):
        # Process Data
        process_data(params, data_int, params.plx["data_count"])
    else:
        # recover self.experiment.data_objects_list = List of the subject names
        preprocessed_data_path = params.plx["save_path"] + params.plx[
            "experiment_uuid"]  # "D:/PhysioNet/processed/sa6pr7/"
        pickle_object = params.plx["experiment_uuid"] + ".pckl"
        subject_folders = [name for name in os.listdir(preprocessed_data_path) if not name == pickle_object]

        relevant_subjects = subject_folders[:train_total]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)

    num_epochs = params.plx.get('epochs')
    apply_oversampling = params.plx.get('apply_oversampling')  # !only on training data

    # MODEL
    timestamps['modelstart'] = datetime.datetime.now()

    if params.plx.get("load_model"):
        model_none, callbacks = choose_model(params, do_compile=True, no_model=True)
        model = load_model("/output/trained_model.hdf5")
    else:
        # build model
        model, callbacks = choose_model(params, do_compile=True)
    timestamps['modelend'] = datetime.datetime.now()

    # TRAINING
    train_count = params.plx.get('train_count')
    val_count = params.plx.get('val_count')

    train_generator = InterIntraEpochGenerator(data_int, params, train_count, start_val=0, shuffle=True,
                                               oversampling=apply_oversampling, crossval_samples=train_total)
    validation_generator = InterIntraEpochGenerator(data_int, params, val_count,
                                                    start_val=train_count, crossval_samples=train_total)

    print("####\n\n\nTraining###\n\n")
    # add weights to prevent overfitting one single classes
    class_weights = train_generator.class_weights

    timestamps['trainingstart'] = datetime.datetime.now()

    history = model.fit_generator(generator=train_generator,
                                  epochs=num_epochs,
                                  callbacks=callbacks,
                                  workers=0,
                                  validation_data=validation_generator,
                                  use_multiprocessing=False,
                                  class_weight=class_weights,
                                  shuffle=True)

    timestamps['trainingend'] = datetime.datetime.now()
    # save model
    model.save("/output/trained_model.hdf5")        # Winslow

    print("Model Training done. Save Performance to Log ... ")

    val_acc_history = history.history['val_accuracy']  # val_accuracy for Winslow, val_acc local
    val_loss_history = history.history['val_loss']

    print("=======> Performance Evaluation <=======")
    train_parameters = count_params(model.trainable_weights)
    record_performance(val_acc_history, val_loss_history, params, timestamps, train_parameters)

    # print evaluation metrics
    evaluation_obj = Eval()
    evaluation_obj.evaluate(params=params,
                            data_int=data_int,
                            model=model)


if __name__ == "__main__":
    train_iitnet_paramtuning()
