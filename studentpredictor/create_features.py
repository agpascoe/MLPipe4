from datetime import datetime, date
import pandas as pd
from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))


def extract_features(df):
    date_stamp = pd.to_datetime(df.published_timestamp)
    df["published_timestamp"] = date_stamp.dt.date
    df["days_since_published"] = (pd.to_datetime(Config.DATE_END_TO_ANALYSIS, utc=True) - date_stamp).dt.days
    return df[["num_lectures", "price", "days_since_published", "content_duration"]]

train_features = extract_features(train_df)
test_features = extract_features(test_df)

train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

train_df.num_subscribers.to_csv(
    str(Config.FEATURES_PATH / "train_labels.csv"), index=None
)
test_df.num_subscribers.to_csv(
    str(Config.FEATURES_PATH / "test_labels.csv"), index=None
)
