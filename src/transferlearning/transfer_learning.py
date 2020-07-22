# @sp: newly created

import os
import sys
import datetime

import keras
from keras.models import load_model
from keras.optimizers import Adam

from models.model_functions import choose_model
from data.DataInterface import DataInterface as DataInt
from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator
from transferlearning.evaluation_transferlearning import EvalTL
from transferlearning.performance_logging import LoggerTL
from keras.utils.vis_utils import plot_model


class TransferLearning:
    """
    Class for applying Transfer Learning methods to the IITNet architecture, using
    source domain physionet and target domain sleep-edf.
    """

    def __init__(self, params, is_winslow):
        self.params = params
        self.model = None
        self.mode = params.plx.get('tl_mode')
        self.stage = params.plx.get('tl_stage')
        self.winslow = is_winslow
        self.load_model = params.plx.get('load_model')
        self.data_int = None
        self.data_int_test = None  # only used for evaluation
        self.path_pretrained_model = ""
        self.path_resulting_model = ""
        self.result_path = ""
        self.base_path = ""

        self.num_epochs = params.plx.get('epochs')

        self.training_history = None
        self.timestamps = dict()
        self.eval_results = dict()
        self.confusion_matrix = dict()
        self.class_report = dict()

        self.__setup()

    def __setup(self):
        # Get the data interface
        # If this is pretraining, use the physionet data, otherwise use sleep-edf data
        if self.winslow:
            if self.mode == "pretrain":
                path_dataset = "/resources/sa6pr7/physionet_challenge/processed/"
            else:
                path_dataset = "/resources/sa6pr7/sleep-edf-v1/sleep-cassette/processed/"
        else:
            if self.mode == "pretrain":
                path_dataset = "D:/physionet_challenge/processed/sa6pr7/"
            elif self.mode == "pretrain_multiple":
                path_dataset = "D:/physionet_shhs1/processed/"
            else:
                path_dataset = "D:/sleep-edf-v1/sleep-cassette/processed/"

        # setup path to folders
        path_training = path_dataset + "training/"
        path_test = path_dataset + "test/"
        # get the numbers of data samples
        data_train_total = self.params.plx.get('train_count') + self.params.plx.get('val_count')
        data_test = self.params.plx.get('test_count')

        # Create Data Interface for the Training Data
        data_int = self.get_data(data_path=path_training, num_samples=data_train_total)
        self.data_int = data_int

        if self.stage == "evaluation":
            # Test data is stored separately to training data --> use separate data interface
            data_int_test = self.get_data(data_path=path_test, num_samples=data_test)
            self.data_int_test = data_int_test

        # Setup paths
        if self.winslow:
            self.base_path = "/output/transfer_learning/"
        else:
            self.base_path = os.getcwd() + "/"

        self.result_path = self.base_path + 'results/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        self.path_pretrained_model = self.base_path + "trained_models/" + "pretrained_model.hdf5"
        self.path_resulting_model = self.result_path + "resulting_model" + ".hdf5"

        os.mkdir(self.result_path)

    def get_data(self, data_path, num_samples):
        # Create the Data Interface
        data_int = DataInt(save_path=data_path,
                           perform_save_raw=self.params.plx["save_raw_data"],
                           key_labels=self.params.plx["key_labels"],
                           uuid=self.params.plx["experiment_uuid"])

        # Recover the data from the repository
        # self.experiment.data_objects_list = List of the subject names
        preprocessed_data_path = data_path + self.params.plx["experiment_uuid"]
        pickle_object = self.params.plx["experiment_uuid"] + ".pckl"
        subject_folders = [name for name in os.listdir(preprocessed_data_path) if not name == pickle_object]

        # Check the number of subjects in the folder
        if len(subject_folders) < num_samples:
            print("Not enough data samples in Repository!")
            print("Requested", str(num_samples), ", Got", len(subject_folders))
            sys.exit()

        relevant_subjects = subject_folders[:num_samples]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)
        return data_int

    def pretrain(self):
        """
        Pretrain the model on the source domain data and save the model.
        """
        # MODEL BUILDING
        if self.load_model:
            self.timestamps['model_start'] = datetime.datetime.now()
            model = load_model(self.path_pretrained_model)
            self.timestamps['model_end'] = datetime.datetime.now()
            # get the callbacks
            model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        else:
            model, callbacks = self.build_model()
        apply_oversampling = self.params.plx.get('apply_oversampling')

        # MODEL TRAINING
        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_pretrained_model)

    def train_whole_model(self):
        """
        Train the whole model from scratch with the target data (train + validation data)
        """
        # MODEL BUILDING
        if self.load_model:
            self.timestamps['model_start'] = datetime.datetime.now()
            model = load_model(self.path_pretrained_model)
            self.timestamps['model_end'] = datetime.datetime.now()
            # get the callbacks
            model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        else:
            model, callbacks = self.build_model()
        apply_oversampling = self.params.plx.get('apply_oversampling')

        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)
        print("Training done.")

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_resulting_model)

    def build_model(self):
        self.timestamps['model_start'] = datetime.datetime.now()
        model, callbacks = choose_model(self.params, do_compile=True)
        self.timestamps['model_end'] = datetime.datetime.now()

        return model, callbacks

    def direct_transfer(self):
        # get the pretrained model from directory
        model = load_model(self.path_pretrained_model)
        self.model = model

    def load_multi_gpu_model(self):
        """
        Since a keras multi-gpu model is used on winslow, a workaround is needed to load the weights.
        See https://github.com/keras-team/keras/issues/11253#issuecomment-482467792 for more information.
        """
        path_model = self.base_path + "trained_models/" + "cpu_model.hdf5"

        multi_model, callbacks = self.build_model()  # Rückgabe: multi-gpu model
        multi_model.load_weights(self.path_pretrained_model)
        single_model = multi_model.layers[-2]
        single_model.save(path_model)

        model = load_model(path_model)
        return model

    def finetuning_dense(self):
        # get pretrained model from directory
        if self.winslow:
            model = self.load_multi_gpu_model()
        else:
            model = load_model(self.path_pretrained_model)

        # freeze every layer, except the last one
        for layer in model.layers[:-1]:
            layer.trainable = False

        # check the trainable status of the layers
        for layer in model.layers:
            print(layer, layer.trainable)

        # get the callbacks
        model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        apply_oversampling = self.params.plx.get('apply_oversampling')

        # set a small learning rate for evaluation and recompile the model
        optimizer = self.get_optimizer()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # train on the target dataset
        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)
        print("Finetuning done.")

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_resulting_model)

    def finetuning_bilstm(self):
        # get pretrained model from directory
        if self.winslow:
            model = self.load_multi_gpu_model()
        else:
            model = load_model(self.path_pretrained_model)

        # freeze every layer, except the last three
        for layer in model.layers[:-3]:
            layer.trainable = False

        # check the trainable status of the layers
        for layer in model.layers:
            print(layer, layer.trainable)

        # get the callbacks
        model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        apply_oversampling = self.params.plx.get('apply_oversampling')

        # set a small learning rate for evaluation and recompile the model
        optimizer = self.get_optimizer()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # train on the target dataset
        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)
        print("Finetuning done.")

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_resulting_model)

    def finetuning_cnn(self):
        # get pretrained model from directory
        if self.winslow:
            model = self.load_multi_gpu_model()
        else:
            model = load_model(self.path_pretrained_model)

        # freeze the last layers, leave the cnn part unfrozen
        for layer in model.layers[-4:]:
            layer.trainable = False

        # check the trainable status of the layers
        for layer in model.layers:
            print(layer, layer.trainable)

        # get the callbacks
        model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        apply_oversampling = self.params.plx.get('apply_oversampling')

        # set a small learning rate for evaluation and recompile the model
        optimizer = self.get_optimizer()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # train on the target dataset
        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)
        print("Finetuning done.")

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_resulting_model)

    def finetuning_complete(self):
        # get pretrained model from directory
        model = load_model(self.path_pretrained_model)

        # don't freeze anything
        # check the trainable status of the layers
        for layer in model.layers:
            print(layer, layer.trainable)

        # get the callbacks
        model_none, callbacks = choose_model(params=self.params, do_compile=True, no_model=True)
        apply_oversampling = self.params.plx.get('apply_oversampling')

        # set a small learning rate for evaluation and recompile the model
        optimizer = self.get_optimizer()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # train on the target dataset
        if self.stage == "training":
            # If in Training stage, separate between training and validation data
            model, history = self.train_model_training(model=model,
                                                       callbacks=callbacks,
                                                       apply_oversampling=apply_oversampling)
        else:
            # For Evaluation stage, train the model with the whole training data
            model, history = self.train_model_evaluation(model=model,
                                                         callbacks=callbacks,
                                                         apply_oversampling=apply_oversampling)
        print("Finetuning done.")

        # Save the Performances and the trained model
        self.training_history = history  # save for later evaluation
        self.model = model
        model.save(self.path_resulting_model)

    def train_model_training(self, model, callbacks, apply_oversampling):
        # If in Training stage, separate between training and validation data
        train_count = self.params.plx.get('train_count')
        val_count = self.params.plx.get('val_count')

        train_generator = InterIntraEpochGenerator(data_int=self.data_int, params=self.params,
                                                   num_subjects=train_count, start_val=0,
                                                   shuffle=True, oversampling=apply_oversampling)
        validation_generator = InterIntraEpochGenerator(data_int=self.data_int, params=self.params,
                                                        num_subjects=val_count, start_val=train_count)

        # add weights to prevent overfitting on some classes
        class_weights = train_generator.class_weights

        print("\n\n\n####TRAINING###\n\n")
        self.timestamps['training_start'] = datetime.datetime.now()
        history = model.fit_generator(generator=train_generator,
                                      epochs=self.num_epochs,
                                      callbacks=callbacks,
                                      validation_data=validation_generator,
                                      workers=0,
                                      use_multiprocessing=False,
                                      class_weight=class_weights,
                                      shuffle=True)
        self.timestamps['training_end'] = datetime.datetime.now()

        return model, history

    def train_model_evaluation(self, model, callbacks, apply_oversampling):
        train_count = self.params.plx.get('train_count') + self.params.plx.get('val_count')

        train_generator = InterIntraEpochGenerator(data_int=self.data_int, params=self.params,
                                                   num_subjects=train_count, start_val=0,
                                                   shuffle=True, oversampling=apply_oversampling)

        # add weights to prevent overfitting on some classes
        class_weights = train_generator.class_weights

        print("\n\n\n####TRAINING###\n\n")
        self.timestamps['training_start'] = datetime.datetime.now()
        history = model.fit_generator(generator=train_generator,
                                      epochs=self.num_epochs,
                                      callbacks=callbacks,
                                      workers=0,
                                      use_multiprocessing=False,
                                      class_weight=class_weights,
                                      shuffle=True)
        self.timestamps['training_end'] = datetime.datetime.now()

        return model, history

    def get_optimizer(self):
        # get the optimizer for the finetuning, with small learning rate
        learning_rate = self.params.plx.get('lr')
        beta1 = self.params.plx.get('beta1')
        beta2 = self.params.plx.get('beta2')

        adam_optimizer = Adam(lr=learning_rate,
                              beta_1=beta1,
                              beta_2=beta2)
        return adam_optimizer

    def evaluate_training(self):
        # evaluate on validation data
        data_int = self.data_int

        num_subjects = self.params.plx.get('val_count')
        offset = self.params.plx.get('train_count')

        self.evaluate(data_int=data_int, num_subjects=num_subjects, start_val=offset)

    def evaluate_test(self):
        # evaluate on test data
        data_int = self.data_int_test

        num_subjects = self.params.plx.get('test_count')
        offset = 0

        self.evaluate(data_int=data_int, num_subjects=num_subjects, start_val=offset)

    def evaluate(self, data_int, num_subjects, start_val):
        # make an evaluation object
        evaluation_obj = EvalTL()

        # evaluate
        cm, accuracy, precision, recall, f_1, class_report = evaluation_obj.evaluate(params=self.params,
                                                                                     data_int=data_int,
                                                                                     model=self.model,
                                                                                     num_subjects=num_subjects,
                                                                                     start_val=start_val)

        # put the global parameters together in a dictionary
        eval_results_global = {'accuracy': accuracy,
                               'precision': precision,
                               'recall': recall,
                               'f_1': f_1}

        # store everything in the object
        self.eval_results = eval_results_global
        self.confusion_matrix = cm
        self.class_report = class_report

    def log_results(self, timestamps):
        all_timestamps = {**self.timestamps, **timestamps}

        # create Logger object
        logger_obj = LoggerTL(params=self.params,
                              training_history=self.training_history,
                              result_path=self.result_path,
                              base_path=self.base_path,
                              cm=self.confusion_matrix,
                              class_report=self.class_report,
                              timestamps=all_timestamps,
                              is_winslow=self.winslow)

        # save all the results to the respository
        logger_obj.log_results()

        # save the keras model graph
        path_img = self.result_path + "keras_model_graph.png"
        #plot_model(model=self.model, to_file=path_img, show_shapes=False, show_layer_names=False)
