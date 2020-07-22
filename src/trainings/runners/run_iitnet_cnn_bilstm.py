# @sp: newly created

import os
import sys

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')

import params.polyaxon_parsing_iitnet_cnn_lstm as pp3

from data.process_data import process_data
from models.model_functions import choose_model
from evaluation.evaluation_obj import Eval
from params.Params_Winslow import Params, winslow_params
from data.DataInterface import DataInterface as DataInt
from trainings.InterIntraEpochGenerator import InterIntraEpochGenerator


def main():
    """
    This is the main workflow for the ml-algorithm
    """

    # get parameters
    params = Params()

    # get additional parameters for iitnet
    plx: dict = pp3.get_parameters()
    params.plx.update(plx)
    # params.plx['batch_size'] = 250
    params.plx['subject_batch'] = 1  # !
    params.plx['apply_downsampling'] = True     # param common_frequency has to be set
    # NOTE: mdl_architecture has to be set to 'iitnet_cnn_bilstm'

    # adjust winslow parameters
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        winslow_params(params)

    # Build model
    model, callbacks = choose_model(params)

    # Get data
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
        preprocessed_data_path = params.plx["save_path"] + params.plx["experiment_uuid"]
        pickle_object = params.plx["experiment_uuid"] + ".pckl"
        subject_folders = [name for name in os.listdir(preprocessed_data_path) if not name == pickle_object]

        relevant_subjects = subject_folders[:train_total]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)

    # Model Training
    print("####\n\n\nTraining###\n\n")
    num_epochs = params.plx.get('epochs')
    apply_oversampling = params.plx.get('apply_oversampling')   # !only on training data

    train_generator = InterIntraEpochGenerator(data_int, params, params.plx.get('train_count'), shuffle=True,
                                               oversampling=apply_oversampling)
    validation_generator = InterIntraEpochGenerator(data_int, params, params.plx.get('val_count'),
                                                    start_val=params.plx['train_count'])

    model.fit_generator(generator=train_generator,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        workers=0,
                        validation_data=validation_generator,
                        use_multiprocessing=False)

    # Model Evaluation
    print("####\n\n\nEvaluation###\n\n")
    evaluation_obj = Eval()
    evaluation_obj.evaluate(params=params,
                            data_int=data_int,
                            model=model)


if __name__ == "__main__":
    main()
