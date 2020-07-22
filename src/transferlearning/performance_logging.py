# @sp: newly created

import datetime
import os
import numpy as np
import json
from csv import DictWriter


class LoggerTL:
    """
    Log all results from Transfer Learning
    """
    def __init__(self, params, training_history, result_path, base_path, cm, class_report, timestamps, is_winslow=False):
        self.params = params

        self.training_history = training_history

        self.confusion_matrix = cm
        self.class_report = class_report

        self.timestamps = timestamps

        self.base_path = base_path
        self.result_path = result_path
        self.winslow = is_winslow

    def log_results(self):
        # save training history --> dict('loss': list, 'acc': list)
        if not self.params.plx.get("tl_mode") == "direct":
            training_history = self.training_history.history
            save_path_history = self.result_path + "training_history.json"
            with open(save_path_history, 'w') as f_obj:
                json.dump(training_history, f_obj, default=convert)

        # save confusion matrix to JSON
        confusion_matrix = self.confusion_matrix.tolist()
        save_path_cm = self.result_path + "confusion_matrix.json"
        with open(save_path_cm, 'w') as f_obj:
            json.dump(confusion_matrix, f_obj)

        # save classwise evaluation to JSON
        classwise_scoring = self.class_report
        save_path_class = self.result_path + "classwise_scoring.json"
        with open(save_path_class, 'w') as f_obj:
            json.dump(classwise_scoring, f_obj)

        # save all parameters to csv
        self.write_logging_csv()

    def write_logging_csv(self):
        path_csv = self.base_path + 'results/' + "logs_all_runs.csv"

        params_dict = self.get_params_dict()
        field_names = params_dict.keys()

        # Replace "." with "," to make the table easily readable in german excel
        for key, value in params_dict.items():
            if isinstance(value, np.float64):
                value = float(value)
            if isinstance(value, float):
                num_as_str = "{:.9f}".format(value)
                num_as_str = num_as_str.replace('.', ',')
                params_dict[key] = num_as_str

        # create a file if it doesn't exist already
        if not os.path.isfile(path_csv):
            with open(path_csv, 'w', newline='') as csvfile:
                fieldnames = field_names
                writer = DictWriter(csvfile, fieldnames=fieldnames, dialect='excel-tab')
                writer.writeheader()

        with open(path_csv, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=field_names, dialect='excel-tab')
            dict_writer.writerow(params_dict)

    def get_params_dict(self):
        """
        Read the parameters to be logged
        """
        params_dict = dict()

        # transfer learning parameters
        params_dict = self.get_tl_params(params_dict)

        # environment variables
        params_dict = self.get_environment_params(params_dict)

        # model building parameters (L, l, overlaps, regularization)
        params_dict = self.get_model_params(params_dict)

        # training parameters (epochs, optimizer values)
        params_dict = self.get_training_params(params_dict)

        # evaluation parameters (timestamps, accuracy, loss, precision, recall, f1)
        params_dict = self.get_timestamps(params_dict)
        params_dict = self.get_scoring_params(params_dict)

        return params_dict

    def get_tl_params(self, params_dict):
        """
        Return Transfer Learning Settings
        """
        params_dict['TL_mode'] = self.params.plx.get('tl_mode')
        params_dict['TL_stage'] = self.params.plx.get('tl_stage')

        return params_dict

    def get_environment_params(self, params_dict):
        """
        Return environment info: Winslow run id, results folder, tensorboard logs
        """
        if self.winslow:
            winslow_id = os.environ['WINSLOW_STAGE_NAME'] + '_' + str(int(os.environ['WINSLOW_STAGE_NUMBER'])+1)
            params_dict['winslow_id'] = winslow_id
        else:
            params_dict['winslow_id'] = None

        params_dict['result_path'] = self.result_path
        if not self.params.plx.get("tl_mode") == "direct":
            params_dict['tb_logs'] = self.params.logdir_tb
        else:
            params_dict['tb_logs'] = None

        return params_dict

    def get_model_params(self, params_dict):
        """
        Return model building parameters: L_epochs, l_subepochs, l_overlapping, filter size, regularization factor
        """
        params_dict['L_epochs'] = self.params.plx.get('l_epochs')
        params_dict['l_subepochs'] = self.params.plx.get('l_subepochs')
        params_dict['l_overlapping'] = self.params.plx.get('l_overlapping')
        params_dict['cnn_filtersize'] = self.params.plx.get('filtersize')
        params_dict['l2_reg_factor'] = self.params.plx.get('l_regularization_factor')

        return params_dict

    def get_training_params(self, params_dict):
        """
        Return model training parameters: batch_size, train_epochs, train/test/val count, oversampling,
                                          optimizer: learning rate, beta1, beta2, epsilon
        """
        params_dict['train_epochs'] = self.params.plx.get('epochs')
        params_dict['batch_size'] = self.params.plx.get('batch_size')
        params_dict['train_samples'] = self.params.plx.get('train_count')
        params_dict['val_samples'] = self.params.plx.get('val_count')
        params_dict['test_samples'] = self.params.plx.get('test_count')
        params_dict['oversampling'] = self.params.plx.get('apply_oversampling')
        params_dict['o_learningrate'] = self.params.plx.get('lr')
        params_dict['o_beta1'] = self.params.plx.get('beta1')
        params_dict['o_beta2'] = self.params.plx.get('beta2')
        params_dict['o_epsilon'] = self.params.plx.get('epsilon')

        return params_dict

    def get_timestamps(self, params_dict):
        """
        Return timestamps and times: processing start, total tl time, model building time,
                                     training time, evaluation time, total time (until now)
        """
        process_start = self.timestamps['processstart']
        params_dict['process_start'] = process_start

        params_dict['tl_time'] = self.timestamps['tl_end'] - self.timestamps['tl_start']

        if self.params.plx.get("tl_mode") == "pretrain" or self.params.plx.get("tl_mode") == "scratch":
            params_dict['model_time'] = self.timestamps['model_end'] - self.timestamps['model_start']
        else:
            params_dict['model_time'] = None

        if not self.params.plx.get("tl_mode") == 'direct':
            params_dict['train_time'] = self.timestamps['training_end'] - self.timestamps['training_start']
        else:
            params_dict['train_time'] = None

        params_dict['eval_time'] = self.timestamps['eval_end'] - self.timestamps['eval_start']

        params_dict['total_time'] = datetime.datetime.now() - process_start

        return params_dict

    def get_scoring_params(self, params_dict):
        """
        Return evaluation scores: Training: Last Accuracy, Last Loss
                                  Evaluation: Accuracy, Precision, Recall, F1-Score
        """
        if not self.params.plx.get("tl_mode") == 'direct':
            train_history = self.training_history.history

            if self.winslow:
                params_dict['train_acc'] = train_history['accuracy'][-1]
            else:
                params_dict['train_acc'] = train_history['acc'][-1]
            params_dict['train_loss'] = train_history['loss'][-1]
        else:
            params_dict['train_acc'] = None
            params_dict['train_loss'] = None

        params_dict['eval_acc'] = self.class_report['accuracy']
        params_dict['eval_precision_macro'] = self.class_report['macro avg']['precision']
        params_dict['eval_recall_macro'] = self.class_report['macro avg']['recall']
        params_dict['eval_f1_macro'] = self.class_report['macro avg']['f1-score']

        return params_dict


def convert(o):
    if isinstance(o, np.float32):
        return float(o)
