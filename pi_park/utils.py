import yaml
import os
import logging
import time

logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, out, m: str):
        self.out = out
        self.m = m

    def __enter__(self):#, out, m: str):
        self.start_time = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.out(self.m.format(time.time() - self.start_time))

def clear_dir(dirpath: str):
    assert os.path.isdir(dirpath), f"Path is not a dir or doesnt exist: {dirpath}"
    files = os.listdir(dirpath)
    for file in files:
        file_path = os.path.join(dirpath, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def load_yaml_file(path):
    if not os.path.exists(path):
        logger.error(f"YAML file at {path} does NOT exist")
        raise FileNotFoundError(f"YAML file at {path} does NOT exist")
    with open(path, mode="r") as f:
        return yaml.safe_load(f)
    
def get_current_time(format="%dd_%Hh_%Mm_%Ss"):
    return time.strftime(format)