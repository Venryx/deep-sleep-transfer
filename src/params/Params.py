import sys
import threading
import time
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
    This enables the code to use the polyaxon

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

    # output paths
    file_path_mdl = define_prepare_mdl_path(plx)
    logdir_tb = define_prepare_tb_path()

