import yaml


def load_yaml_file(path="./configs/basic_config.yml"):
    with open(path, mode="r") as f:
        return yaml.safe_load(f)