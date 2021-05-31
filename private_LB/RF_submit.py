# Random Forest
# for private LB
# should be run in /Solar_Predict

import pickle
import pandas as pd

from privateLB import API, date_ctrl

KEY = "sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=="


def predict(location, base_date, base_time):
    # get data
    data = API(base_date, base_time, location, KEY).get_data(
        preprocess=True, itp_method="quadratic"
    )

    # load model
    with open(f"witt_modeling/rf_models/{location}_model.pkl", "rb") as f:
        model = pickle.load(f)

    # predict
    x = data.loc[:, X_COLS]
    predicted = model.predict(x)
    return predicted


def to_submission(path, predict_date, predicted):
    submission = pd.read_csv("./private_LB/test_sample_submission.csv")
    # Doesn't have to be 'ulsan'. Can be any arbitrary column.
    submission.loc[submission["time"].str.contains(predict_date), "ulsan"] = predicted
    submission.to_csv(path, index=False)


def tweak_after_prediction(array, N):
    # replace by 320 if value < 320
    # add N if value > 320
    for i in range(len(array)):
        if array[i] < 320:
            array[i] = 320
        else:
            array[i] += N
    return array


def main(base_date, base_time):
    # path to save submission file
    PATH = f"./witt_modeling/privateLB_submissions/rf_{base_date}-{base_time}_submission.csv"

    # sum dangjin prediction and ulsan prediction
    dj = predict("dangjin", base_date, base_time)
    uls = predict("ulsan", base_date, base_time)
    total = dj + uls

    # add value to compensate underestimation
    total = tweak_after_prediction(total, N)
    print(total)

    to_submission(PATH, date_ctrl(base_date, SHIFT, "pandas"), total)


# constants
SHIFT = 1
X_COLS = [
    "Temperature",
    "Humidity",
    "Cloud",
    "Day_cos",
    "Day_sin",
    "Year_cos",
    "Year_sin",
]
N = 20

if __name__ == "__main__":
    main("20210531", "2000")
