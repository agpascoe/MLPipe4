#import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
import yaml

params = yaml.safe_load(open("params.yaml"))["create_dataset"]

np.random.seed(params["SEED"])

Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

#gdown.download(
#    "https://drive.google.com/uc?id=1gkYBOIMm8pAGunRoI3OzQHQrgOdaRjfC",
#    str(Config.ORIGINAL_DATASET_FILE_PATH),
#)

df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))

df_train, df_test = train_test_split(
    df, test_size=Config.SPLIT_DATASET, random_state=Config.RANDOM_SEED,
)

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)
