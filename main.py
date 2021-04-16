# ======================================================================== #
# ========================== From gitlab Clone =========================== #
# ======================================================================== #

'''
- ìì  ìê°: 21.04.16 19:00
- ë´ì©
1) gitlab cloneì ìëí  ì ìëë¡ ìì 
2) dangjin, ulsan dataì ëí xgboost model gridsearch ì§í
'''

import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob

# path
path = "original_dataset/"
dangjin_fcst_data_path = path + "dangjin_fcst_data.csv"
dangjin_obs_data_path = path + "dangjin_obs_data.csv"
energy_data_path = path + "energy.csv"
ulsan_fcst_data_path = path + "ulsan_fcst_data.csv"
ulsan_obs_data_path = path + "ulsan_obs_data.csv"
site_info_path = path + "site_info.csv"

# file convert to pandas data
dangjin_fcst_data = pd.read_csv(dangjin_fcst_data_path)
dangjin_obs_data = pd.read_csv(dangjin_obs_data_path)
energy_data = pd.read_csv(energy_data_path)
ulsan_fcst_data = pd.read_csv(ulsan_fcst_data_path)
ulsan_obs_data = pd.read_csv(ulsan_obs_data_path)
site_info = pd.read_csv(site_info_path)

'''
- ì¼ì, ìê° ë±ì 'time' ì´ë¼ë ë³ìëªì¼ë¡ íµì¼
- ê·¸ ì¸ì ë³ìëªì 'ë³ìëª_íì¼ëª' ì¼ë¡ ì§ì (ulsan, dangjin)
- ê°ì íê¸¸ ulsan ë°ì´í°ì dangjin ë°ì´í°ë ìë¡ ë¬´ê´íë¤. 
- ì´íì NAN ë°ì´í°ë dropna()ë¥¼ ì´ì©í´ ëª¨ë ìê±° ìì 
- íëì ë°ì´í°ìì êµ¬ì¶í ë¤, ìê´ê´ê² ë¶ì + ADF test ë±ì ì§í
'''

dangjin_fcst_data.rename(
    columns = {
        "Forecast time":"time", 
        "forecast":"forecast_fcst", 
        "Temperature":"temp_fcst",
        "Humidity":"humid_fcst",
        "WindSpeed":"ws_fcst",
        "WindDirection":"wd_fcst",
        "Cloud":"cloud_fcst"
        }, inplace = True)

dangjin_obs_data.rename(
    columns = {
        "ì¼ì":"time",
        "ê¸°ì¨(Â°C)":"temp_obs",
        "íì(m/s)":"ws_obs",
        "íí¥(16ë°©ì)":"wd_obs",
        "ìµë(%)":"humid_obs",
        "ì ì´ë(10ë¶ì)":"cloud_obs"
    }, inplace = True)

ulsan_fcst_data.rename(
    columns = {
        "Forecast time":"time", 
        "forecast":"forecast_fcst", 
        "Temperature":"temp_fcst",
        "Humidity":"humid_fcst",
        "WindSpeed":"ws_fcst",
        "WindDirection":"wd_fcst",
        "Cloud":"cloud_fcst"
    }, inplace = True)

ulsan_obs_data.rename(
    columns = {
        "ì¼ì":"time",
        "ê¸°ì¨(Â°C)":"temp_obs",
        "íì(m/s)":"ws_obs",
        "íí¥(16ë°©ì)":"wd_obs",
        "ìµë(%)":"humid_obs",
        "ì ì´ë(10ë¶ì)":"cloud_obs"
    }, inplace = True)

dangjin_obs_data = dangjin_obs_data.drop(columns = ["ì§ì ", "ì§ì ëª"])
ulsan_obs_data = ulsan_obs_data.drop(columns = ["ì§ì ","ì§ì ëª"])

# fcst_data ë°ì´í° ì ì²ë¦¬
# time + forecast -> timeì¼ë¡ ì í, ì´í ì¤ë³µëë ê°ì íê·  ì²ë¦¬ 

dangjin_fcst_data["time_fcst"] = pd.to_datetime(dangjin_fcst_data["time"].copy()) + dangjin_fcst_data["forecast_fcst"].copy().astype("timedelta64[h]")
dangjin_fcst_data = dangjin_fcst_data.groupby("time_fcst", as_index = False).mean()
dangjin_fcst_data = dangjin_fcst_data.drop(columns = ["forecast_fcst"])

ulsan_fcst_data["time_fcst"] = pd.to_datetime(ulsan_fcst_data["time"].copy()) + ulsan_fcst_data["forecast_fcst"].copy().astype("timedelta64[h]")
ulsan_fcst_data = ulsan_fcst_data.groupby("time_fcst", as_index = False).mean()
ulsan_fcst_data = ulsan_fcst_data.drop(columns = ["forecast_fcst"])

ulsan_fcst_data.rename(columns = {"time_fcst":"time"}, inplace = True)
dangjin_fcst_data.rename(columns = {"time_fcst":"time"}, inplace = True)

ulsan_fcst_data = ulsan_fcst_data.astype({"time":"object"})
dangjin_fcst_data = dangjin_fcst_data.astype({"time":"object"})

dangjin_obs_data["time"] = pd.to_datetime(dangjin_obs_data["time"].copy(), format='%Y-%m-%d %H:%M:%S')
dangjin_obs_data = dangjin_obs_data.astype({"time":"object"})

ulsan_obs_data["time"] = pd.to_datetime(ulsan_obs_data["time"].copy(), format='%Y-%m-%d %H:%M:%S')
ulsan_obs_data = ulsan_obs_data.astype({"time":"object"})

# energy_dataë time í­ëª©ì´ stringì¼ë¡ ì ì¥ëì´ ìë¤. ì´ë¥¼ timestampë¡ ì²ë¦¬í´ì¼íë¤. 

energy_data_time_tmp = energy_data["time"].copy()

for i in range(energy_data.shape[0]):
    if energy_data["time"][i][-8:] == "24:00:00":
        energy_data["time"][i] = energy_data_time_tmp[i].replace("24:00:00", " 00:00:00")

    energy_data["time"][i] = pd.Timestamp(energy_data["time"][i])

energy_data = energy_data.astype({"time":"object"})

# ì  ë°ì´í° NAN ì²ë¦¬

dangjin_fcst_data = dangjin_fcst_data.dropna()
dangjin_obs_data = dangjin_obs_data.dropna()
energy_data = energy_data.dropna()
ulsan_fcst_data = ulsan_fcst_data.dropna()
ulsan_obs_data = ulsan_obs_data.dropna()

# fcst_data['time'] time interval: 3hour -> 1hourë¡ ì¶ì íì
# Lagrangian Interpolation

def interpolation(df):

    df_copy = df.copy()
    var_names = df.columns

    total_s = list()
    time_list = list()
    
    for var_name in var_names:
        s = list()
        for i in range(df_copy.shape[0] - 1):
            timedeltas = df_copy["time"][i+1] - df_copy["time"][i]
            n_intervals = int(timedeltas / np.timedelta64(1, "h"))

            for j in range(n_intervals):
        
                if var_name == "time":
                    time_stamps = df_copy["time"][i] + timedeltas * j / n_intervals
                    time_list.append(time_stamps)
                else:
                    add_ = df_copy[var_name][i] + (df_copy[var_name][i+1] - df_copy[var_name][i]) / n_intervals * j
                    s.append(add_)

        if var_name == "time":
            time_list = np.array(time_list).reshape(-1,1)
            total_s.append(time_list)
        else:
            s = np.array(s).reshape(-1,1)
            total_s.append(s)

    total_s = np.array(total_s).T.reshape(-1, len(var_names))
    df_converted = pd.DataFrame(total_s, columns = var_names)

    return df_converted

dangjin_fcst_data = interpolation(dangjin_fcst_data.copy())
ulsan_fcst_data = interpolation(ulsan_fcst_data.copy())

ulsan_fcst_data = ulsan_fcst_data.astype({"time":"object"})
dangjin_fcst_data = dangjin_fcst_data.astype({"time":"object"})

# total dataset êµ¬ì±

from functools import reduce

list_dangjin = [dangjin_fcst_data, dangjin_obs_data, energy_data[["time","dangjin_floating","dangjin_warehouse","dangjin"]].copy()]
list_ulsan = [ulsan_fcst_data, ulsan_obs_data, energy_data[["time","ulsan"]].copy()]

dangjin_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_dangjin)
ulsan_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_ulsan)

display(dangjin_data)
display(ulsan_data)

'''
- dangjin ë° ulsan ì§ì­ ë´ ëª¨ë  ì¸¡ì /ìì¸¡ ë³ìë¥¼ íëì ë°ì´í°ìì í¬í¨
- ì¤ë³µëë ìê°ëì ëí´ìë§ ì ë³ í íëì ë°ì´í°ìì¼ë¡ íµí©
'''

# EDA 
import seaborn as sns
from scipy import stats

# stats.pearsonrì ì´ì©í´ ê° ë³ìê° pearson correlationì êµ¬íë¤. 
# ì´í dangjinê³¼ ulsan datasetsì ëí heatmapì ê·¸ë¦°ë¤. 

var_names = dangjin_data.copy().drop(columns = ["dangjin_floating","dangjin_warehouse", "dangjin", "time"]).columns

print("statistical analysis with dangjin_floating: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin_floating"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")
print("statistical analysis with dangjin_warehouse: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin_warehouse"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")
print("statistical analysis with dangjin: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")

dangjin_data.corr("pearson")
sns.heatmap(dangjin_data.corr("pearson"))

# ulsan data

var_names = ulsan_data.copy().drop(columns = ["ulsan", "time"]).columns

print("statistical analysis with ulsan: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(ulsan_data[var_name], ulsan_data["ulsan"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")

ulsan_data.corr("pearson")
sns.heatmap(ulsan_data.corr("pearson"))

# supervised learningì ìí preprocessing

def series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = False):

    '''
    - function: to convert series data to be supervised 
    - data: pd.DataFrame
    - x_name: the name of variables used to predict
    - y_name: the name of variables for prediction
    - n_in: number(or interval) of series used to predict
    - n_out: number of series for prediction

    - 24 * 30 -> 720ê°ì outputì ìì¸¡
    - íìí input -> ìµì 720ê° ì´ì
    - ìì´ëì´: 1ì¼ ìì¸¡, ìì¸¡ì¹ë¥¼ ë¤ì ìë ¥ê°ì¼ë¡ ë°ê² ì§í, ì´ ê²½ì° output:24

    '''

    data_copy = data.copy()
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        #cols.append(data_copy[x_name].shift(i)) # col: [data_copy.shift(n_in), .... data_copy.shift(1)]
        cols.append(data_copy[x_name].shift(i))
        names += [("%s(t-%d)"%(name, i)) for name in x_name]
    
    for i in range(0, n_out):
        y = data_copy[y_name]
        cols.append(y.shift(-i))
        # cols:[data_copy.shift(n_in-1), .... data_copy.shift(1), data_copy[y_name].shift(0)....data_copy[y_name].shift(-n_out + 1)]

        if i == 0:
            names += [("%s(t)"%(name)) for name in y_name]
        else:
            names += [("%s(t+%d)"%(name, i)) for name in y_name]

    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace = True)
    
    return agg

# model architecture

!sudo pip install xgboost
!sudo pip install --upgrade xgboost

import xgboost as xgb

'''
- dangjin, ulsan ë ì§ì­ì ëí ë°ì ë ìì¸¡ ìê³ ë¦¬ì¦
- gridsearchCVë¥¼ ì´ì©í parameter tunning ìë
'''

# dangjin
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating", "dangjin_warehouse", "dangjin"]
y_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating", "dangjin_warehouse", "dangjin"]
n_features = len(x_name)
n_in = 24 * 3
n_out = 1
ratio = 0.8 # train size ratio

dangjin_data_supervised = series_to_supervised(dangjin_data, x_name, y_name, n_in, n_out, dropnan = True)

x_data = dangjin_data_supervised.values[:,:-6]
y_data = dangjin_data_supervised.values[:,-6:]

# train_test split
data_size = x_data.shape[0]
train_size = int(data_size * ratio)

x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_train = y_data[0:train_size]
y_test = y_data[train_size:]

# build model
params = {
    "booster":"gbtree",
    "objective":"reg:squarederror",
    "learning_rate":0.01,
    "max_depth":15,
    "n_estimator":1000,
    "nthread":-1,
    "gamma":0.0,
    "min_child_weight":1
}

# gridsearch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    "eta":[0.001, 0.01, 0.1, 0.3],
    "booster":["gbtree", "linear"],
    "objective":["reg:squarederror", "reg:linear"],
    "learning_rate":[0.01, 0.001],
    "max_depth":[3, 5, 10, 15, 20],
    "n_estimators":[50, 100, 200, 500, 1000],
    "nthread":[-1],
    "gamma":[0, 0.01, 0.1, 1.0, 10, 100],
    "subsample":[0.2, 0.5, 0.8],
    "colsample_bytree":[0.2, 0.5, 0.8],
    "colsample_bylevel":[0.2, 0.5, 0.8],
    "min_child_weight":[1,3,5],
    "reg_lambda":[0.01, 0.1, 1.0, 10],
    "reg_alpha":[0, 0.1, 1.0]
}

cv = KFold(n_splits = 5, random_state = 42)
model = xgb.XGBRegressor()

grid = GridSearchCV(model, param_grid = param_grid, cv = cv, scoring = "neg_root_mean_squared_error", n_jobs = -1, refit = True)

# model_params for dangjin_floating
grid.fit(x_train, y_train[:,3].reshape(-1,1))
print("optimized parameters for dangjin_floating: ",grid.best_params_)

# model_params for dangjin_warehouse
grid.fit(x_train, y_train[:,4].reshape(-1,1))
print("optimized parameters for dangjin_warehouse: ",grid.best_params_)

# model_params for dangjin
grid.fit(x_train, y_train[:,5].reshape(-1,1))
print("optimized parameters for dangjin: ",grid.best_params_)

# build model and parameter setting

def build_xgb(params):
    model = xgb.XGBRegressor(
        booster = params["booster"],
        objective = params["objective"],
        learning_rate = params["learning_rate"],    
        max_depth = params["max_depth"],
        n_estimators = params["n_estimators"],
        nthread = params["nthread"],
        gamma = params["gamma"],
        min_child_weight = params["min_child_weight"],
        subsample = params["subsample"],
        reg_lambda = params["reg_lambda"],
        reg_alpha = params["reg_alpha"],
        colsample_bytree = params["colsample_bytree"],
        colsample_bylevel = params["colsample_bylevel"]
    )
    return model

dangjin_models = [build_xgb(params) for i in range(n_features)]

# train model
for i in range(n_features):
    var_name_predict = y_name[i]
    y_var_train = y_train[:,i].reshape(-1,1)
    y_var_test = y_test[:,i].reshape(-1,1)
    print("forecasting variable:%s"%(var_name_predict))
    dangjin_models[i].fit(x_train, y_var_train, early_stopping_rounds = 400, eval_set = [(x_test, y_var_test)])

# submission_prediction

def submission_predict(models, x_data, n_predict):
    '''
    - for multi-varaible prediction
    - x_data:(timesteps, n_features)
    - n_predict: timesteps for forecasting
    - x_data_after: x_data[1:] + predict_value
    '''
    total_prediction = None
    y_preds = []
    x_data_after = x_data

    for i in range(n_predict):
        y_preds = []
        for j in range(n_features):
            y_pred = models[j].predict(x_data_after)
            y_preds.append(y_pred)

        y_preds = np.array(y_preds).reshape(1,-1)
        x_data_after = np.concatenate((x_data_after, y_preds), axis = 1)[0,n_features:].reshape(1,-1)
        

        if total_prediction is None:
            total_prediction = y_preds
        else:
            total_prediction = np.concatenate((total_prediction, y_preds), axis = 0)
    
    total_prediction = total_prediction.reshape(-1, n_features)

    return total_prediction

# test
input_prediction = dangjin_data[x_name][-24*30*1 - n_in:-24*30*1].values.reshape(1,-1)
actual = dangjin_data[y_name][-24*30*1:].values
prediction = submission_predict(dangjin_models, input_prediction, n_predict = 24 * 30)

for i in range(len(y_name)):
    var_name = y_name[i]
    plt.figure(i)  
    
    yreal = actual[:,i]
    yhat = prediction[:,i]
    plt.title(var_name + "prediction")
    plt.ylabel(var_name)
    plt.plot(yreal, "r", label = "actual value")
    plt.plot(yhat, "b", label = "forecasting")
    plt.legend()
    plt.show()

# ulsan
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
n_features = len(x_name)
n_in = 24 * 3
n_out = 1
ratio = 0.8 # train size ratio

ulsan_data_supervised = series_to_supervised(ulsan_data, x_name, y_name, n_in, n_out, dropnan = True)

x_data = ulsan_data_supervised.values[:,:-4]
y_data = ulsan_data_supervised.values[:,-4:]

# train_test split
data_size = x_data.shape[0]
train_size = int(data_size * ratio)

x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_train = y_data[0:train_size]
y_test = y_data[train_size:]

# build model

params = {
    "booster":"dart",
    "objective":"reg:squarederror",
    "learning_rate":0.001,
    "max_depth":15,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":10.0,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "colsample_bylevel":0.8,
    "min_child_weight":3,
    "reg_lambda":1.0,
    "reg_alpha":0.1
}

ulsan_models = [build_xgb(params) for i in range(n_features)]

# train model
for i in range(n_features):
    var_name_predict = y_name[i]
    y_var_train = y_train[:,i].reshape(-1,1)
    y_var_test = y_test[:,i].reshape(-1,1)
    print("forecasting variable:%s"%(var_name_predict))
    ulsan_models[i].fit(x_train, y_var_train, early_stopping_rounds = 400, eval_set = [(x_test, y_var_test)])


# submission_prediction for ulsan

def submission_predict(models, x_data, n_predict):
    '''
    - for multi-varaible prediction
    - x_data:(timesteps, n_features)
    - n_predict: timesteps for forecasting
    - x_data_after: x_data[1:] + predict_value
    '''
    total_prediction = None
    y_preds = []
    x_data_after = x_data

    for i in range(n_predict):
        y_preds = []
        for j in range(n_features):
            y_pred = models[j].predict(x_data_after)
            y_preds.append(y_pred)

        y_preds = np.array(y_preds).reshape(1,-1)
        x_data_after = np.concatenate((x_data_after, y_preds), axis = 1)[0,n_features:].reshape(1,-1)
        

        if total_prediction is None:
            total_prediction = y_preds
        else:
            total_prediction = np.concatenate((total_prediction, y_preds), axis = 0)
    
    total_prediction = total_prediction.reshape(-1, n_features)

    return total_prediction


# test
input_prediction = ulsan_data[x_name][-24*30*1 - n_in:-24*30*1].values.reshape(1,-1)
actual = ulsan_data[y_name][-24*30*1:].values
prediction = submission_predict(ulsan_models, input_prediction, n_predict = 24 * 30)

for i in range(len(y_name)):
    var_name = y_name[i]
    plt.figure(i)  
    
    yreal = actual[:,i]
    yhat = prediction[:,i]
    plt.title(var_name + "prediction")
    plt.ylabel(var_name)
    plt.plot(yreal, "r", label = "actual value")
    plt.plot(yhat, "b", label = "forecasting")
    plt.legend()
    plt.show()


# NMAE-10 ì§í í¨ì

def sola_nmae(answer_df, submission_df):
    submission = submission_df[submission_df['time'].isin(answer_df['time'])]
    submission.index = range(submission.shape[0])
    
    # ìê°ëë³ ì´ ë°ì ë
    sum_submission = submission.iloc[:,1:].sum(axis=1)
    sum_answer = answer_df.iloc[:,1:].sum(axis=1)
    
    # ë°ì ì ë°ì ì©ë
    capacity = {
        'dangjin_floating':1000, # ë¹ì§ììíìê´ ë°ì ì©ë
        'dangjin_warehouse':700, # ë¹ì§ìì¬ì°½ê³ íìê´ ë°ì ì©ë
        'dangjin':1000, # ë¹ì§íìê´ ë°ì ì©ë
        'ulsan':500 # ì¸ì°íìê´ ë°ì ì©ë
    }
    
    # ì´ ë°ì ì©ë
    total_capacity = np.sum(list(capacity.values()))
    
    # ì´ ë°ì ì©ë ì ëì¤ì°¨
    absolute_error = (sum_answer - sum_submission).abs()
    
    # ë°ì ì©ëì¼ë¡ ì ê·í
    absolute_error /= total_capacity
    
    # ì´ ë°ì ì©ëì 10% ì´ì ë°ì í ë°ì´í° ì¸ë±ì¤ ì¶ì¶
    target_idx = sum_answer[sum_answer>=total_capacity*0.1].index
    
    # NMAE(%)
    nmae = 100 * absolute_error[target_idx].mean()
    
    return nmae
