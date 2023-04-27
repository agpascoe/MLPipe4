import json
import math
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error

from config import Config
import mlflow

mlflow.set_experiment(experiment_name=Config.EXPERIMENT_NAME)

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))

with mlflow.start_run(run_name="testing"):    
    r_squared = model.score(X_test, y_test)
    mlflow.log_metric('score',r_squared)

y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(r_squared=r_squared, rmse=rmse), outfile)

mlflow.end_run()
