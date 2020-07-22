# @sp: newly created

import os
import sys

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')
sys.path.append('/resources/')

import params.polyaxon_parsing_iitnet_cnn_lstm as pp3

from keras.models import load_model
from data.process_data import process_data
from evaluation.evaluation_obj import Eval
from params.Params_Winslow import Params, winslow_params
from data.DataInterface import DataInterface as DataInt


def evaluate_iitnet():
    """
    Evaluate the trained IITNet model
    """

    # Setup the parameters
    params = Params()

    # get additional parameters for iitnet
    plx: dict = pp3.get_parameters()
    params.plx.update(plx)

    # adjust winslow parameters
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        winslow_params(params)

    params.plx['subject_batch'] = 1  # !
    # NOTE: mdl_architecture has to be set to 'iitnet_cnn_bilstm'

    # Setup and load data
    # change the save path to get the Test Data
    params.plx["save_path"] = "/resources/sa6pr7/physionet_challenge/processed/test/"    # Winslow
                            # "D:/physionet_challenge/processed/sa6pr7/training/"       #local
    num_test_data = params.plx.get("test_count")

    data_int = DataInt(save_path=params.plx["save_path"],
                       perform_save_raw=params.plx["save_raw_data"],
                       key_labels=params.plx["key_labels"],
                       uuid=params.plx["experiment_uuid"])

    if not params.plx.get("data_already_processed"):
        # Process Data
        process_data(params, data_int, num_test_data)
    else:
        # recover self.experiment.data_objects_list = List of the subject names
        preprocessed_data_path = params.plx["save_path"] + params.plx["experiment_uuid"]
        pickle_object = params.plx["experiment_uuid"] + ".pckl"
        subject_folders = [name for name in os.listdir(preprocessed_data_path) if not name == pickle_object]

        relevant_subjects = subject_folders[:num_test_data]
        data_int.experiment.recover_data_objectlist(relevant_subjects)

        print("Data already processed. Recover", str(len(relevant_subjects)), "Subjects from", preprocessed_data_path)

    # Load the trained model
    print("Load trained model from ", params.file_path_mdl, " ... ", end=" ")
    model = load_model(params.file_path_mdl)
    print("done")

    # Model Evaluation
    print("####\n\n\nEvaluation###\n\n")
    evaluation_obj = Eval()
    evaluation_obj.evaluate(params=params,
                            data_int=data_int,
                            model=model)


if __name__ == "__main__":
    evaluate_iitnet()
