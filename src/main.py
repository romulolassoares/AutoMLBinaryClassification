from modules.automl import Automl
import pandas as pd
import yaml
from utils.create_folders import create_dirs

config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

create_dirs(config)

file = config["tests"]["dataset_file"]

df = pd.read_csv(file)

automl = Automl(df, file, config=config)
top_models = automl.find_best_models(4)

tuned_models = []
for model in top_models:
    tuned_model = automl.tuning_model(model)
    tuned_models.append(tuned_model)

model_evaluations = []
for model in tuned_models:
    model_evaluation = automl.eval_model(model)
    model_evaluations.append(model_evaluation)
    
#automl.save_model(model)

for evaluation in model_evaluations:
    print(f"{evaluation["model_name"]}: {evaluation["accuracy"]}")

#forma de identificar overfiting
# diferença da accuracy entr test e train
# se o percentual for maior que uma porcentagem 10%
# possivel caso

#tem outra forma melhor????