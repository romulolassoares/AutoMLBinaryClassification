import os
import yaml


def create_dirs_from_yaml(config_path: str) -> None:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    base_path = config["paths"].get("base", "")
    for key, sub_path in config["paths"].items():
        if key == "base":
            continue
        full_path = os.path.join(base_path, sub_path)
        os.makedirs(full_path, exist_ok=True)


create_dirs_from_yaml("config.yaml")
