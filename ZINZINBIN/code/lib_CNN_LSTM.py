import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import glob


# ======================================================================== #
# ================== build model and training(CNN_LSTM) ================== #
# ======================================================================== #

# CNN_LSTM model 

class CNN_LSTM(tf.keras.models.Model):
    def __init__(self, filters = 64, kernel_size = 3, strides = 1, pool_size = 2, dropout = 0.2, units = 128, n_predict = 24):
        super(CNN_LSTM, self).__init__()
        # batch_normalize
        self.batch1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        # Conv1
        self.conv1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "valid", kernel_initializer = "glorot_uniform", activation = "relu", name = "conv1", kernel_regularizer= tf.keras.regularizers.l2(0.001))
        )
         # Conv2
        self.conv2 =  tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(filters = filters*2, kernel_size = kernel_size, strides = strides, padding = "valid", kernel_initializer = "glorot_uniform", activation = "relu", name = "conv2", kernel_regularizer= tf.keras.regularizers.l2(0.001))
        )
        self.batch2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        # connection between CNN and RNN
        self.avgpool = tf.keras.layers.TimeDistributed(
            tf.keras.layers.AveragePooling1D(pool_size = pool_size)
        )

        self.dropout1 = tf.keras.layers.Dropout(dropout)

        # Conv3
        self.conv3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(filters = filters*2, kernel_size = kernel_size, strides = strides, padding = "valid", kernel_initializer = "glorot_uniform", activation = "relu", name = "conv1", kernel_regularizer= tf.keras.regularizers.l2(0.001))
        )       
        # Conv4
        self.conv4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(filters = filters*4, kernel_size = kernel_size, strides = strides, padding = "valid", kernel_initializer = "glorot_uniform", activation = "relu", name = "conv1", kernel_regularizer= tf.keras.regularizers.l2(0.001))
        )          
        self.batch2_ = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.avgpool_ = tf.keras.layers.TimeDistributed(
            tf.keras.layers.AveragePooling1D(pool_size = pool_size)
        )

        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.flatten = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )
        

        # RNN
        self.lstm1 = tf.keras.layers.LSTM(units = units, activation = "tanh", recurrent_activation = "tanh", return_sequences = False, kernel_regularizer= tf.keras.regularizers.l2(0.01))
        self.repeat = tf.keras.layers.RepeatVector(n_predict)
        self.lstm2 = tf.keras.layers.LSTM(units = units, activation = "tanh", recurrent_activation = "tanh", return_sequences = True, kernel_regularizer= tf.keras.regularizers.l2(0.01))

        self.batch3 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        # Regression
        self.dense_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(64, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))
        )

        self.batch4 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.dense_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(64, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))
        )
        self.output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, name = "output")
        )

    def predict(self, x):
        # 값을 호출하기 위한 목적
        x = self.batch1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.avgpool(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch2_(x)
        x = self.avgpool_(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.lstm1(x)
        x = self.repeat(x)
        x = self.lstm2(x)
        x = self.batch3(x)
        x = self.dense_1(x)
        x = self.batch4(x)
        x = self.dense_2(x)
        outputs = self.output_layer(x)

        return outputs

    def call(self, x):
        x = self.batch1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.avgpool(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch2_(x)
        x = self.avgpool_(x)   
        x = self.dropout2(x)     
        x = self.flatten(x)
        x = self.lstm1(x)
        x = self.repeat(x)
        x = self.lstm2(x)
        x = self.batch3(x)
        x = self.dense_1(x)
        x = self.batch4(x)
        x = self.dense_2(x)
        outputs = self.output_layer(x)

        return outputs

def build_CNN_LSTM(input_shape, filters, kernel_size, strides, pool_size, dropout, units, n_predict):
    '''
    use: build_CNN_LSTM(input_shape, filters, kernel_size, strides, pool_size, dropout, units, n_predict)
    - input_shape: (1, n_timesteps, n_features)
    - filters: CNN filters
    - kernel_size: CNN kernel
    - strides: CNN strides
    - pool_size: AveragePooling1D pool_size
    - dropout: Dropout layer ration
    - units: LSTM unit cell 
    - output: (Batch_size, n_predict, 1)
    '''
    inputs = tf.keras.layers.Input(shape = input_shape, name = "input_layer")
    outputs = CNN_LSTM(
        filters = filters,
        kernel_size = kernel_size, 
        strides = strides, 
        pool_size = pool_size, 
        dropout = dropout, 
        units = units, 
        n_predict = n_predict)(inputs)
    model = tf.keras.models.Model(inputs, outputs, name = "CNN_LSTM")
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.RMSprop(lr = 1e-3),
        run_eagerly = True
    )
    model.summary()

    return model

def series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = False):

    # x_name: Temp...etc except PG
    # y_name: PG
    # n_in and n_out: equal

    data_copy = data.copy()
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        #cols.append(data_copy[x_name].shift(i)) # col: [data_copy.shift(n_in), .... data_copy.shift(1)]
        cols.append(data_copy[x_name].shift(i))
        names += [("%s(t-%d)"%(name, i)) for name in x_name]
    
    for i in range(n_out, 0, -1):
        y = data_copy[y_name]
        cols.append(y.shift(i))
        # cols:[data_copy.shift(n_in-1), .... data_copy.shift(1), data_copy[y_name].shift(0)....data_copy[y_name].shift(-n_out + 1)]

        names += [("%s(t-%d)"%(name, i)) for name in y_name]

    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace = True)
    
    return agg


def preprocess_wind(data):
    '''
    data: pd.DataFrmae which contains the columns 'WindSpeed' and 'WindDirection'
    '''

    # degree to radian
    wind_direction_radian = data['WindDirection'] * np.pi / 180

    # polar coordinate to cartesian coordinate
    wind_x = data['WindSpeed'] * np.cos(wind_direction_radian)
    wind_y = data['WindDirection'] * np.sin(wind_direction_radian)

    # name pd.series
    wind_x.name = 'Wind_X'
    wind_y.name = 'Wind_Y'

    return wind_x, wind_y

# add seasonality
def day_of_year(datetime): #pd.datetime
    return pd.Period(datetime, freq='D').day_of_year

def add_seasonality(df):
    new_df = df.copy()
    
    new_df['Day_cos'] = new_df['time'].apply(lambda x: np.cos(x.hour * (2 * np.pi) / 24))
    new_df['Day_sin'] = new_df['time'].apply(lambda x: np.sin(x.hour * (2 * np.pi) / 24))

    new_df['Year_cos'] = new_df['time'].apply(lambda x: np.cos(day_of_year(x) * (2 * np.pi) / 365))
    new_df['Year_sin'] = new_df['time'].apply(lambda x: np.sin(day_of_year(x) * (2 * np.pi) / 365))

    return new_df