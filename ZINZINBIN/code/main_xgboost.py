'''
# ======================================================================== #
# =========================== File Explanation =========================== #
# ======================================================================== #

- model: xgboost
- preprocessing: dropna, (wd, ws) -> (ws_x, ws_y), month, day, hour -> (m,d,h)
- file structure: main, lib_function
- datasets: original datasets + 2015 - 2018 datasets(external)
'''

# library
import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from lib_function_0518 import *
import pickle

# ======================================================================== #
# ===================== load data and preprocessing ====================== #
# ======================================================================== #

# load data
with open('./witt_preprocessing/pickles/dangjin_merged.pkl','rb') as f:
    dangjin_data = pickle.load(f)
with open('./witt_preprocessing/pickles/ulsan_merged.pkl','rb') as f:
    ulsan_data = pickle.load(f)

# energy = 0 drop
dangjin_data = dangjin_data[dangjin_data["dangjin"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_floating"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_warehouse"] != 0]
ulsan_data = ulsan_data[ulsan_data["ulsan"] != 0]

# ======================================================================== #
# ================== build model and training(xgboost) =================== #
# ======================================================================== #

# model architecture
import xgboost as xgb
esr = 400 # early stopping round

params = {
    "booster":"dart",
    #"objective":"reg:pseudohubererror",
    "objective":"reg:squarederror",
    "learning_rate":0.1,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1024,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":0.1,
    "reg_alpha":0.1, # 1.0
    "sample_type":"uniform"
}


# ulsan 
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["ulsan"]
ulsan_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(ulsan_data.iloc[0:-24*30], x_name, y_name, test_size = 0.3)
custom_eval_ulsan = lambda x,y : custom_evaluation(x, y, cap = "ulsan")

ulsan_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_ulsan)
xgb.plot_importance(ulsan_model, height = 0.9)

# dangjin floating

params = {
    "booster":"dart",
    #"objective":"reg:pseudohubererror",
    "objective":"reg:squarederror",
    "learning_rate":0.1,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1024,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":0.1,
    "reg_alpha":0.1, # 1.0
    "sample_type":"uniform"
}

x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin_floating"]
dangjin_floating_model = build_xgb(params)
dangjin_data = dangjin_data.dropna()

x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.3)
custom_eval_dangjin_floating = lambda x,y : custom_evaluation(x, y, cap = "dangjin_floating")
dangjin_floating_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_floating)
xgb.plot_importance(dangjin_floating_model, height = 0.9)

# dangjin warehouse
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin_warehouse"]
dangjin_warehouse_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.3)
custom_eval_dangjin_warehouse = lambda x,y : custom_evaluation(x, y, cap = "dangjin_warehouse")
dangjin_warehouse_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_warehouse)
xgb.plot_importance(dangjin_warehouse_model, height = 0.9)

# dangjin 
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin"]
dangjin_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.3)
custom_eval_dangjin = lambda x,y : custom_evaluation(x, y, cap = "dangjin")
dangjin_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin)
xgb.plot_importance(dangjin_model, height = 0.9)


# ======================================================================== #
# ================= forecasting and evaluate the model =================== #
# ======================================================================== #

# evaluation
term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

# ulsan evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["ulsan"]

n_predict = 24 * 30
fcst_data = ulsan_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))


yhat = submission_predict_xgb(ulsan_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = ulsan_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: ulsan"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("ulsan, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

ulsan_nmae = sola_nmae(yreal, yhat, cap = "ulsan")
print("nmae for ulsan: ", ulsan_nmae)

# dangjin_floating evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin_floating"]

n_in = 1
n_predict = 24 * 30
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

yhat = submission_predict_xgb(dangjin_floating_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_floating"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_floating, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

dangjin_floating_nmae = sola_nmae(yreal, yhat, cap = "dangjin_floating")
print("nmae for dangjin_floating: ", dangjin_floating_nmae)


# dangjin_warehouse evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin_warehouse"]

n_predict = 24 * 30
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

yhat = submission_predict_xgb(dangjin_warehouse_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_warehouse"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_warehouse, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()
    
dangjin_warehouse_nmae = sola_nmae(yreal, yhat, cap = "dangjin_warehouse")
print("nmae for dangjin_warehouse: ", dangjin_warehouse_nmae)


# dangjin evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
y_name = ["dangjin"]

n_in = 1
n_predict = 24 * 30
start_data_in = dangjin_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

yhat = submission_predict_xgb(dangjin_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

dangjin_nmae = sola_nmae(yreal, yhat, cap = "dangjin")
print("nmae for dangjin: ", dangjin_nmae)

'''

# submission 
submission_path = "./submission.csv"
submission = pd.read_csv(submission_path)


ulsan_obs_feb_path = "./original_dataset/external_data/ulsan_obs_2021-02.csv" 
dangjin_obs_feb_path = "./original_dataset/external_data/dangjin_obs_2021-02.csv"

ulsan_obs_feb = pd.read_csv(ulsan_obs_feb_path, encoding = "CP949" ) 
dangjin_obs_feb = pd.read_csv(dangjin_obs_feb_path, encoding = "CP949")

dangjin_obs_feb.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

ulsan_obs_feb.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

dangjin_obs_feb = dangjin_obs_feb.drop(columns = ["지점", "지점명"])
ulsan_obs_feb = ulsan_obs_feb.drop(columns = ["지점","지점명"])

def preprocess_wind(data):

    # degree to radian
    wind_direction_radian = data['WindDirection'] * np.pi / 180

    # polar coordinate to cartesian coordinate
    wind_x = data['WindSpeed'] * np.cos(wind_direction_radian)
    wind_y = data['WindDirection'] * np.sin(wind_direction_radian)

    # name pd.series
    wind_x.name = 'Wind_X'
    wind_y.name = 'Wind_Y'

    return wind_x, wind_y

dangjin_obs_feb = dangjin_obs_feb.join(preprocess_wind(dangjin_obs_feb))
ulsan_obs_feb = ulsan_obs_feb.join(preprocess_wind(ulsan_obs_feb))

for i in range(dangjin_obs_feb.shape[0]):
    dangjin_obs_feb["time"][i] = pd.Timestamp(dangjin_obs_feb["time"][i])
    
for i in range(ulsan_obs_feb.shape[0]):
    ulsan_obs_feb["time"][i] = pd.Timestamp(ulsan_obs_feb["time"][i])
    
dangjin_obs_feb = dangjin_obs_feb.astype({"time":"object"})
ulsan_obs_feb = ulsan_obs_feb.astype({"time":"object"})
    
# month, day, hour addition for obs_feb
month = []
day = []
hour = []

for i in range(len(dangjin_obs_feb)):
    month.append(dangjin_obs_feb["time"][i].month)
    day.append(dangjin_obs_feb["time"][i].day)
    hour.append(dangjin_obs_feb["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

dangjin_obs_feb["month"] = month
dangjin_obs_feb["day"] = day
dangjin_obs_feb["hour"] = hour

dangjin_obs_feb["month"] = dangjin_obs_feb["month"].astype(int)
dangjin_obs_feb["day"] = dangjin_obs_feb["day"].astype(int)
dangjin_obs_feb["hour"] = dangjin_obs_feb["hour"].astype(int)


month = []
day = []
hour = []

for i in range(len(ulsan_obs_feb)):
    month.append(ulsan_obs_feb["time"][i].month)
    day.append(ulsan_obs_feb["time"][i].day)
    hour.append(ulsan_obs_feb["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

ulsan_obs_feb["month"] = month
ulsan_obs_feb["day"] = day
ulsan_obs_feb["hour"] = hour

ulsan_obs_feb["month"] = ulsan_obs_feb["month"].astype(int)
ulsan_obs_feb["day"] = ulsan_obs_feb["day"].astype(int)
ulsan_obs_feb["hour"] = ulsan_obs_feb["hour"].astype(int)

# ulsan forecasting
x_name_fcst = ["month","day","hour","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]


obs_data = ulsan_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
yhat = submission_predict_xgb(ulsan_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,4] = yhat

# dangjin_floating forecasting
x_name_fcst = ["month","day","hour","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
yhat = submission_predict_xgb(dangjin_floating_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,1] = yhat

# dangjin_warehouse forecasting
x_name_fcst = ["month","day","hour","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
yhat = submission_predict_xgb(dangjin_warehouse_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,2] = yhat

# dangjin forecasting
x_name_fcst = ["month","day","hour","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
yhat = submission_predict_xgb(dangjin_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,3] = yhat

submission.to_csv("submission.csv", index = False)
'''