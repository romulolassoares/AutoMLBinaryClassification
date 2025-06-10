import os

def create_dirs(config: dict) -> None:
    base_path = config["paths"].get("base", "")
    for key, sub_path in config["paths"].items():
        if key == "base":
            continue
        full_path = os.path.join(base_path, sub_path)
        os.makedirs(full_path, exist_ok=True)