import os

def is_running_locally():
    """
    Tells if an experiment is running on a local machine or not
    :return:
    """
    local = False
    if "POLYAXON_NO_OP" in os.environ:
        local = True
    return local


def get_username():
    username = os.getlogin()
    return username