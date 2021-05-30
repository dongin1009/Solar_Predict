# Random Forest
# for private LB
# should be run in /Solar_Predict

import pickle
import pandas as pd

from privateLB import API, date_ctrl

KEY = "sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=="


def predict(location):
    # get data
    data = API(BASE_DATE, BASE_TIME, location, KEY).get_data()

    # load model
    with open(f"witt_modeling/rf_models/{location}_model.pkl", "rb") as f:
        model = pickle.load(f)

    # predict
    x = data.loc[:, X_COLS]
    predicted = model.predict(x)
    return predicted


def to_submission(path, predict_date, predicted):
    submission = pd.read_csv("original_dataset/sample_submission.csv")
    submission.loc[submission["time"].str.contains(predict_date), "ulsan"] = predicted
    submission.to_csv(path, index=False)


BASE_DATE = "20210525"
BASE_TIME = "2000"
SHIFT = 20  # 1 in real private LB case
X_COLS = [
    "Temperature",
    "Humidity",
    "Cloud",
    "Day_cos",
    "Day_sin",
    "Year_cos",
    "Year_sin",
]
PATH = f"{BASE_DATE}-{BASE_TIME}_submission"

dj = predict("dangjin")
uls = predict("ulsan")
total = dj + uls

to_submission(PATH, date_ctrl(BASE_DATE, SHIFT, "pandas"), total)

