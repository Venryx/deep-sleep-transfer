# @sp: newly created

import datetime
import os
import sys

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


def train_iitnet_allsubjects():
    """
    !NOT USED ANYMORE! !RUN ON WINSLOW WITH MORE RAM TO USE >200 SUBJECTS!

    Problem: the model training cant handle more than 200 subjects at a time (less with a bigger model).
    So every epoch, the model is trained on 200 subjects first, saved, reloaded and then
    trained on another 200 subjects, etc.
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

    # define number of training subjects
    train_count = params.plx.get('train_count')
    val_count = params.plx.get('val_count')
    total_count = train_count + val_count

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

        relevant_subjects = subject_folders[:total_count]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)

    num_epochs = params.plx.get('epochs')
    apply_oversampling = params.plx.get('apply_oversampling')  # !only on training data

    # build model
    timestamps['modelstart'] = datetime.datetime.now()
    model, callbacks = choose_model(params, do_compile=False)
    timestamps['modelend'] = datetime.datetime.now()
    # save untrained model
    model = compile_model_iitnet(params=params, model=model)
    print("Save untrained model ... ", end=" ")
    model_save_path = params.file_path_raw_mdl
    model.save(model_save_path)
    print("done")

    timestamps_trainingstart = []
    timestamps_trainingend = []
    all_val_accs = []
    all_val_loss = []
    timestamps['crossval_start'] = datetime.datetime.now()

    # split the training data
    total_training_runs = int((train_count // 200) + 1)
    train_per_run = int(train_count // total_training_runs)
    validation_per_run = int(val_count // total_training_runs)

    for training_run in range(total_training_runs):
        # train on max. 200 subjects, evaluate on validation_per_run subjects

        # load the model
        print("Load model ... ", end=" ")
        model = load_model(model_save_path)
        print("done.")

        # set indices
        train_start = training_run * train_per_run
        train_end = train_start + train_per_run
        val_start = train_end

        train_generator = InterIntraEpochGenerator(data_int, params, train_per_run, start_val=train_start,
                                                   shuffle=True, oversampling=apply_oversampling)
        validation_generator = InterIntraEpochGenerator(data_int, params, validation_per_run,
                                                        start_val=val_start)

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

        print("Saving model ... ", end=" ")
        model.save(model_save_path)
        print('done.')

    print("Model Training done. Save Performance to Log ... ", end=" ")

    # log the performance
    val_acc_history = history.history['val_accuracy']        # val_accuracy for Winslow, val_acc local
    val_loss_history = history.history['val_loss']

    all_val_accs.append(val_acc_history)
    all_val_loss.append(val_loss_history)
    print("done.")

    print("=======> Logging Performance Evaluation <=======")
    timestamps['crossval_end'] = datetime.datetime.now()
    timestamps['trainstarts'] = timestamps_trainingstart
    timestamps['trainends'] = timestamps_trainingend
    record_performance(all_val_accs, all_val_loss, params, timestamps)


if __name__ == "__main__":
    train_iitnet_allsubjects()
