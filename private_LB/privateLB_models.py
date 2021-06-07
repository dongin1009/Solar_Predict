# for private LB
# should be run in /Solar_Predict/private_LB

import pickle
import pandas as pd
from glob import glob
import joblib

from privateLB_API import API, date_ctrl


class RfPredict:
    def __init__(self, base_date, base_time):
        # params
        self.base_date = base_date
        self.base_time = base_time
        self.read_path = sorted(glob("./submissions/*.csv"))[-1]  # most recent file

        # constants
        self.X_COLS = ["Temperature", "Humidity", "Cloud", "Day_cos", "Day_sin", "Year_cos", "Year_sin"]
        self.N = 20
        self.KEY = "sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=="

    def _vanilla_predict(self, location, gap):
        # get data
        data = API(self.base_date, self.base_time, location, self.KEY).get_data(
            gap=gap, preprocess=True, itp_method="quadratic"
        )

        # load model
        with open(f"../witt_modeling/rf_models/{location}_model.pkl", "rb") as f:
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

        # file
        predict_date = date_ctrl(self.base_date, gap, "pandas")
        file.loc[file["time"].str.contains(predict_date), "ulsan"] = total

        return file

    def get_submission(self):
        # gap = 1
        submission = pd.read_csv(self.read_path, encoding="euc-kr")
        submission = self._predict(1, submission)

        # gap = 2
        submission = self._predict(2, submission)

        return submission


class LgbmPredict(RfPredict):
    def __init__(self, base_date, base_time):
        super().__init__(base_date, base_time)

        self.X_COLS = ["Day_cos", "Day_sin", "Year_cos", "Year_sin",
                       "Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

        self.MODEL_NAME = 'lgbm'

    def _vanilla_predict(self, location, gap):
        # get data
        data = API(self.base_date, self.base_time, location, self.KEY).get_data(
            gap=gap, preprocess=True, itp_method="linear")

        # load model
        model = joblib.load(f"../ZINZINBIN/trained_model/{location}_{self.MODEL_NAME}.pkl")

        # predict
        x = data.loc[:, self.X_COLS]
        predicted = model.predict(x)
        return predicted

    def _predict(self, gap, file):
        # sum dangjin prediction and ulsan prediction
        dj = self._vanilla_predict("dangjin", gap)
        uls = self._vanilla_predict("ulsan", gap)
        total = dj + uls

        # file
        predict_date = date_ctrl(self.base_date, gap, "pandas")
        file.loc[file["time"].str.contains(predict_date), "ulsan"] = total

        return file

    def get_submission(self):
        return super().get_submission()


class XgbPredict(LgbmPredict):
    def __init__(self, base_date, base_time):
        super().__init__(base_date, base_time)
        self.MODEL_NAME = 'xgb'

    def _vanilla_predict(self, location, gap):
        return super()._vanilla_predict(location, gap)

    def _predict(self, gap, file):
        return super()._predict(gap, file)

    def get_submission(self):
        return super().get_submission()

    