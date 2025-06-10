from modules.automl import Automl
import pandas as pd
import yaml

config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

file = config["tests"]["dataset_file"]

df = pd.read_csv(file)

automl = Automl(df, file)
top_models = automl.find_best_models(2)
tuned_model = automl.tuning_model(top_models[0])
automl.eval_model(tuned_model)
automl.save_model(tuned_model)

