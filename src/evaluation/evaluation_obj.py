import numpy as np

from models.model_functions import calc_steps
from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator     # @sp - add generator for IITNet
from util import model_evaluation_utils as meu
# @sp - remove unnecessary imports


class Eval:
    def __init__(self):
        self.predicted_labels = np.array([])
        self.true_labels = np.array([], dtype=int)

    def evaluate(self, params, data_int, model):
        encoder = data_int.get_encoder()
        experiment = params.experiment

        # >>> @sp - add generator for IITNet
        if params.plx.get('mdl_architecture') == "iitnet_cnn_bilstm":
            eval_generator = InterIntraEpochGenerator(data_int=data_int,
                                                      params=params,
                                                      num_subjects=params.plx["test_count"],
                                                      start_val=params.plx['train_count'] +
                                                                params.plx['val_count'],
                                                      evaluation=True)
        # <<< @sp
        else:
            pass    # @sp - remove all other cases except IITNet

        self.predicted_labels = model.predict_generator(
            eval_generator,
            workers=0,
            use_multiprocessing=False)

        self.predicted_labels = self.decode_labels(self.predicted_labels,
                                                   encoder)

        self.true_labels = np.concatenate(eval_generator.label_list, axis=0)
        self.true_labels = self.decode_labels(self.true_labels, encoder)

        cm, accuracy, precision, recall, f_1 = \
            meu.display_model_performance_metrics(true_labels=self.true_labels,
                                                  predicted_labels=self.predicted_labels,
                                                  classes=encoder.classes_)

        experiment.log_metrics(accuracy=accuracy, precision=precision,
                               recall=recall, f_1_score=f_1)
        print()

    def add_labels(self, labels):
        n = len(labels)
        labels = np.argmax(labels, axis=1)
        for i in labels:
            self.true_labels = np.append(self.true_labels, int(i))

        print(f"{n} labels added")
        print()

    def decode_labels(self, labels, encoder):
        labels = np.asarray([x for x in labels])        # @sp - bugfix
        numbered_labels = np.argmax(labels, axis=1)
        decoded_labels = encoder.inverse_transform(numbered_labels)
        return decoded_labels
