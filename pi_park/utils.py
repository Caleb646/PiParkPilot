import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_yaml_file(path="./configs/basic_config.yml"):
    if not os.path.exists(path):
        logger.error(f"YAML file at {path} does NOT exist")
        raise FileNotFoundError(f"YAML file at {path} does NOT exist")
    with open(path, mode="r") as f:
        return yaml.safe_load(f)