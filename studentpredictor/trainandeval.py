import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import mlflow
import math
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

from sklearn.metrics import mean_squared_error

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

mlflow.set_experiment(experiment_name=Config.EXPERIMENT_NAME)
mlflow.sklearn.autolog()
#model = LinearRegression()
model = RandomForestRegressor(
    n_estimators=150, max_depth=6, random_state=Config.RANDOM_SEED
)
with mlflow.start_run(run_name="testing"):
    model = model.fit(X_train, y_train.to_numpy().ravel())
    r_squared = model.score(X_test, y_test)
    mlflow.log_metric('score',r_squared)
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric('rmse',rmse)
    

mlflow.end_run()
