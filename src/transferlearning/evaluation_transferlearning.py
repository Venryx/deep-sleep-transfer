# @sp: newly created

import numpy as np

from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator
from util import model_evaluation_utils as meu


class EvalTL:
    """
    Evaluation of a Model trained with transfer learning.
    Mostly leaning on evaluation obj, to be integrated later.
    """

    def __init__(self):
        self.predicted_labels = np.array([])
        self.true_labels = np.array([], dtype=int)

    def evaluate(self, params, data_int, model, num_subjects, start_val):
        encoder = data_int.get_encoder()
        experiment = params.experiment

        # define the input data (validation or test data)
        eval_generator = InterIntraEpochGenerator(data_int=data_int,
                                                  params=params,
                                                  num_subjects=num_subjects,
                                                  start_val=start_val,
                                                  evaluation=True,
                                                  shuffle=False,
                                                  oversampling=False)

        # get the labels predicted by the model and decode them
        self.predicted_labels = model.predict_generator(eval_generator,
                                                        workers=0,
                                                        use_multiprocessing=False)
        self.predicted_labels = self.decode_labels(self.predicted_labels,
                                                   encoder)

        # get the correct labels
        self.true_labels = np.concatenate(eval_generator.label_list, axis=0)
        self.true_labels = self.decode_labels(self.true_labels, encoder)

        # get the evaluation parameters
        cm, accuracy, precision, recall, f_1, class_report = \
            meu.display_model_performance_metrics(true_labels=self.true_labels,
                                                  predicted_labels=self.predicted_labels,
                                                  classes=encoder.classes_,
                                                  return_dict=True)

        # return the parameters
        return cm, accuracy, precision, recall, f_1, class_report

    def decode_labels(self, labels, encoder):
        labels = np.asarray([x for x in labels])
        numbered_labels = np.argmax(labels, axis=1)
        decoded_labels = encoder.inverse_transform(numbered_labels)
        return decoded_labels
