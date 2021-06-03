# Random Forest
# for private LB
# should be run in /Solar_Predict

from datetime import date
import pickle
import pandas as pd
from glob import glob

from pandas.core.indexes import base

from privateLB import API, date_ctrl


class RfPredict:
    def __init__(self, base_date, base_time):
        # params
        self.base_date = base_date
        self.base_time = base_time

        # constants
        self.X_COLS = [
            "Temperature",
            "Humidity",
            "Cloud",
            "Day_cos",
            "Day_sin",
            "Year_cos",
            "Year_sin",
        ]
        self.N = 20
        self.KEY = "sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=="

        # execute main
        self.main()

    def _vanilla_predict(self, location, gap):
        # get data
        data = API(self.base_date, self.base_time, location, self.KEY).get_data(
            gap=gap, preprocess=True, itp_method="quadratic"
        )

        # load model
        with open(f"witt_modeling/rf_models/{location}_model.pkl", "rb") as f:
            model = pickle.load(f)

        # predict
        x = data.loc[:, self.X_COLS]
        predicted = model.predict(x)
        return predicted

    def _tweak_after_prediction(self, array):
        # replace by 320 if value < 320
        # add N if value > 320
        for i in range(len(array)):
            if array[i] < 320:
                array[i] = 320
            else:
                array[i] += self.N
        return array

    def _predict(self, gap, file):
        # sum dangjin prediction and ulsan prediction
        dj = self._vanilla_predict("dangjin", gap)
        uls = self._vanilla_predict("ulsan", gap)
        total = dj + uls

        # add value to compensate underestimation
        total = self._tweak_after_prediction(total)
        print(total)

        # file
        predict_date = date_ctrl(self.base_date, gap, "pandas")
        file.loc[file["time"].str.contains(predict_date), "ulsan"] = total

        return file

    def main(self):
        # the most recent file in submission folder
        READ_PATH = sorted(glob("witt_modeling/privateLB_submissions/*.csv"))[-1]
        # path to save submission file
        WRITE_PATH = f"./witt_modeling/privateLB_submissions/rf_{self.base_date}-{self.base_time}_submission.csv"

        # gap = 1
        submission = pd.read_csv(READ_PATH, encoding="euc-kr")
        submission = self._predict(1, submission)

        # gap = 2
        submission = self._predict(2, submission)
        submission.to_csv(WRITE_PATH, index=False)


if __name__ == "__main__":
    RfPredict("20210602", "2000")
