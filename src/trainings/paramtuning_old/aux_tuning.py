# @sp: newly created

import os
from csv import DictWriter
import numpy as np


def record_performance(accs, losses, params, timestamps, train_parameters):
    """
    Record the performance of model training into an csv-file.
    :param history:
    """

    print("Save Performances to Logfile ... ", end=" ")

    params_dict = get_param_dict(accs, losses, params, timestamps, train_parameters)
    field_names = params_dict.keys()

    # save the performance values
    file_name = "/output/performance_paramtuning.csv"

    # create a file if it doesn't exist already
    if not os.path.isfile(file_name):
        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = field_names
            writer = DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')
            writer.writeheader()

    with open(file_name, 'a+', newline='') as write_obj:
        dict_writer = DictWriter(write_obj, fieldnames=field_names, dialect='excel-tab')
        dict_writer.writerow(params_dict)
    print("done.")


def get_param_dict(accs, losses, params, timestamps, train_parameters):
    """
    Get the parameters for the log file and save to dict
    """

    param_dict = {}

    epochs = params.plx.get('epochs')
    param_dict = get_performance_values(param_dict=param_dict, accs=accs, losses=losses, epochs=epochs)
    param_dict = get_hyperparams(param_dict=param_dict, params=params)
    param_dict = get_times(param_dict=param_dict, timestamps=timestamps, epochs=epochs)

    # add the number of trainable parameters in the network
    param_dict['trainable_parameters'] = train_parameters

    return param_dict


def get_performance_values(param_dict, accs, losses, epochs):
    # accuracy and loss

    # get the epoch with the best accuracy
    average_val_accs = np.asarray(accs)
    average_val_loss = np.asarray(losses)
    best_epoch = np.argmax(average_val_accs)

    best_val_acc = average_val_accs[best_epoch]
    best_val_loss = average_val_loss[best_epoch]

    best_epoch += 1
    param_dict['val_accuracy'] = best_val_acc
    param_dict['val_loss'] = best_val_loss
    param_dict['best_epoch'] = best_epoch

    print("Best Validation Accuracy: ", str(best_val_acc))
    print("Best Validation Loss: ", str(best_val_loss))

    return param_dict


def get_hyperparams(param_dict, params):
    """
    Hyperparameters set in Winslow
    """
    param_dict['epochs'] = params.plx.get('epochs')
    param_dict['batch_size'] = params.plx.get('batch_size')
    param_dict['samples'] = params.plx.get('train_count') + params.plx.get('val_count')
    param_dict['l_epochs'] = params.plx.get('l_epochs')
    param_dict['l_subepochs'] = params.plx.get('l_subepochs')
    param_dict['overlapping_subepochs'] = params.plx.get('l_overlapping')
    param_dict['cnn1_filtersize'] = params.plx.get('filtersize')
    param_dict['k_folds_crossval'] = params.plx.get('k_crossval')
    param_dict['o_learningrate'] = params.plx.get('lr')
    param_dict['o_beta1'] = params.plx.get('beta1')
    param_dict['o_beta2'] = params.plx.get('beta2')
    param_dict['o_epsilon'] = params.plx.get('epsilon')
    param_dict['l2_reg_factor'] = params.plx.get('l_regularization_factor')
    param_dict['oversampling'] = params.plx.get('apply_oversampling')

    return param_dict


def get_times(param_dict, timestamps, epochs):
    if 'WINSLOW_STAGE_ID' in os.environ:
        param_dict['winslow_id'] = os.environ['WINSLOW_STAGE_ID']
        param_dict['winslow_startup'] = os.environ['WINSLOW_SETUP_DATE_TIME']

        # time before start of python script = setup-time   # TODO: check how to convert from str to datetime
        # setup_time_sec = param_dict['winslow_startup'] - timestamps['processstart']
        # param_dict['time_setup'] = str(setup_time_sec)

    # model building time
    model_time = timestamps['modelend'] - timestamps['modelstart']
    param_dict['time_modelbuilding'] = model_time.seconds       # doesn't take days into account

    # training time
    train_time = timestamps['trainingend'] - timestamps['trainingstart']
    param_dict['time_training'] = train_time.seconds

    param_dict['mean_time_fold'] = 0

    return param_dict
