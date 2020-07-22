import os
from datetime import datetime

from polyaxon_client.tracking import get_outputs_path


def define_prepare_tb_path():
    logdir_tb = os.path.join(".", "tf_logs", "scalars") # ".\\tf_logs\\scalars\\"
    outputs_path = get_outputs_path()
    if outputs_path is not None:  # polyaxon behavior
        logdir_tb = outputs_path + "/" + logdir_tb
    else:  # local behavior
        logdir_tb = logdir_tb + datetime.now().strftime("%Y%m%d-%H%M%S")
    return logdir_tb


def define_prepare_mdl_path(plx):
    logdir_mdl = "mdl_chkpts/"
    outputs_path = get_outputs_path()
    if outputs_path is not None:  # polyaxon behavior
        logdir_mdl = outputs_path + "/" + logdir_mdl
    if not os.path.exists(logdir_mdl):
        try:
            os.mkdir(logdir_mdl)
        except OSError:
            print("Creation of the directory %s failed" % logdir_mdl)
        else:
            print("Successfully created the directory %s " % logdir_mdl)
    file_path_mdl = logdir_mdl + plx.get('mdl_architecture') + '_' + plx.get('eng_kind') + ".hdf5"

    # >>> @sp - add untrained model path
    file_path_raw_mdl = logdir_mdl + plx.get('mdl_architecture') + '_' + 'untrained' + ".hdf5"
    return file_path_mdl, file_path_raw_mdl
    # <<< @sp
