# @sp: newly created

import sys
import os
import threading
import time
from datetime import datetime
from polyaxon_client.tracking import Experiment

from params import param_utils
import params.polyaxon_parsing as pp
from util.output_artifact_utils import define_prepare_mdl_path, \
    define_prepare_tb_path


def get_file_inputs():
    while True:
        try:
            sys.argv.append(input())
        except EOFError:
            break


class Params:
    """
    Description
    ----
    This enables the code to use winslow. Most of this is copied from Params (for Polyaxon).
    """
    # This is to load the params from a file
    input_thread = threading.Thread(target=get_file_inputs, args=(), daemon=True)
    input_thread.start()
    print("Fetching inputs", end=" ... -> ")
    time.sleep(10)
    print("done.")


    temporal_context = 0
    last_interval = None

    # polyaxon params
    experiment = Experiment()

    plx = pp.get_parameters()
    param_utils.set_params(plx)
    param_utils.check_params(plx)

    # if the environment is within winslow
    if 'WINSLOW_PIPELINE_NAME' in os.environ:
        # output paths
        log_dir_mdl = "/workspace/mdl_chkpts/"
        if not os.path.exists(log_dir_mdl):
            os.mkdir(log_dir_mdl)
            print("Directory ", log_dir_mdl, " Created ")
        else:
            print("Directory ", log_dir_mdl, " already exists")
        file_path_mdl = "/workspace/mdl_chkpts/" + plx.get('mdl_architecture') + '_' + plx.get('eng_kind') + ".hdf5"

        logdir_tb = "/workspace/tf_logs/scalars" + datetime.now().strftime("%Y%m%d-%H%M%S")

        file_path_raw_mdl = "/workspace/mdl_chkpts/" + plx.get('mdl_architecture') + '_' + 'untrained' + ".hdf5"

    else:
        # output paths
        file_path_mdl, file_path_raw_mdl = define_prepare_mdl_path(plx)
        logdir_tb = define_prepare_tb_path()


def winslow_params(params, transferlearning=False):
    """
    If running on winslow, adjust the parameters according to winslow settings
    :param transferlearning:
    :param params:
    """
    if transferlearning == True:
        params.plx['tl_stage'] = os.environ['tl_stage']
        params.plx['tl_mode'] = os.environ['tl_mode']
    else:
        params.plx['k_crossval'] = int(os.environ['k_crossval'])

    params.plx['load_model'] = bool(int(os.environ['load_model']))
    params.plx['apply_oversampling'] = bool(int(os.environ['apply_oversampling']))
    params.plx['epochs'] = int(os.environ['epochs'])
    params.plx['batch_size'] = int(os.environ['batch_size'])
    params.plx['train_count'] = int(os.environ['train_count'])
    params.plx['val_count'] = int(os.environ['val_count'])
    params.plx['test_count'] = int(os.environ['test_count'])
    params.plx['l_epochs'] = int(os.environ['l_epochs'])
    params.plx['l_subepochs'] = int(os.environ['l_subepochs'])
    params.plx['l_overlapping'] = int(os.environ['l_overlapping'])
    params.plx['filtersize'] = int(os.environ['filtersize'])
    params.plx['lr'] = float(os.environ['o_learning_rate'])
    params.plx['beta1'] = float(os.environ['o_beta1'])
    params.plx['beta2'] = float(os.environ['o_beta2'])
    params.plx['epsilon'] = float(os.environ['o_epsilon'])
    params.plx['l_regularization_factor'] = float(os.environ['l_regularization_factor'])
