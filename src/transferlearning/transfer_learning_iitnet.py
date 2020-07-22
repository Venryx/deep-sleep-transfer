# @sp: newly created

import os
import sys

sys.path.append(os.getcwd())  # puts all uploaded python modules into the python path
sys.path.append('/input/src/')
sys.path.append('/resources/')

from transferlearning.transfer_learning import *
import params.polyaxon_parsing_iitnet_cnn_lstm as pp3
from params.Params_Winslow import Params, winslow_params
from keras import backend as K


def main():

    # TIMESTAMPS
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
    is_winslow = False
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        is_winslow = True
        winslow_params(params=params, transferlearning=True)
    else:
        params.plx['l_regularization_factor'] = params.plx.get('l2_regularization_factor')

    params.plx['subject_batch'] = 1  # !
    # NOTE: mdl_architecture has to be set to 'iitnet_cnn_bilstm'

    # TRANSFER LEARNING
    # Create a Transfer Learning object
    tl_obj = TransferLearning(params=params, is_winslow=is_winslow)

    # select Transfer Learning mode
    tl_mode = params.plx.get("tl_mode")
    timestamps['tl_start'] = datetime.datetime.now()

    # select corresponding method
    if tl_mode == "pretrain" or tl_mode == "pretrain_multiple":
        tl_obj.pretrain()
    elif tl_mode == "scratch":
        tl_obj.train_whole_model()
    elif tl_mode == "direct":
        tl_obj.direct_transfer()
    elif tl_mode == "dense":
        tl_obj.finetuning_dense()
    elif tl_mode == "recurrent":
        tl_obj.finetuning_bilstm()
    elif tl_mode == "cnn":
        tl_obj.finetuning_cnn()
    elif tl_mode == "complete":
        tl_obj.finetuning_complete()

    timestamps['tl_end'] = datetime.datetime.now()
    timestamps['eval_start'] = datetime.datetime.now()

    # EVALUATION
    print("\n\n\n####EVALUATION###\n\n")
    if params.plx.get('tl_stage') == "training":
        tl_obj.evaluate_training()
    else:
        tl_obj.evaluate_test()

    timestamps['eval_end'] = datetime.datetime.now()

    # LOGGING
    print("\n\nSaving Results ... ", end=" ")
    tl_obj.log_results(timestamps=timestamps)
    print("done.")

    print("\n\nTransfer Learning finished.")


if __name__ == "__main__":
    main()
