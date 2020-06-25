# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import glob
from sklearn.preprocessing import MinMaxScaler
from network import TSPredictor
from util_functions import aucPerformance

def read_dataset(file_name, _normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=None)
    abnormal_data = abnormal['value'].values
    abnormal_label = abnormal['is_anomaly'].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)
    return abnormal_data, abnormal_label

def plot_ts(file_name, start=None, end=None, zoom=False, _normalize=False):
    
    ts_1 = pd.read_csv(file_name)
    ts_1["timestamp"] = ts_1.index
    if _normalize==True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_1["value"] = scaler.fit_transform(ts_1["value"].values[:, np.newaxis])[:, 0]
    
    if not zoom :
        fig, axes = plt.subplots(1, 1)
        
        ts_1.plot(x="timestamp", y="value",
                ax=axes)
        ts_1.loc[ts_1.is_anomaly == 1].plot(x="timestamp", y="value",
                                            ax=axes, kind="scatter",
                                            c="black")
        axes.legend(["Normal", "anomalies"])
    else :
        fig, axes = plt.subplots(2,1)
        
        ts_1.plot(x="timestamp", y="value",
                ax=axes[0])
        ts_1.loc[ts_1.is_anomaly == 1].plot(x="timestamp", y="value",
                                            ax=axes[0], kind="scatter",
                                            c="black")
        axes[0].legend(["Normal", "anomalies"])
        ts_1 = ts_1[start-1:end]
        ts_1.loc[ts_1.is_anomaly == 1].plot(x="timestamp", y="value",
                                            ax=axes[1], kind="scatter",  c="black")
        axes[1].legend(["anomalies"])
    return fig, axes
    
def partition_ts(ts, label, length=10, future_step=1, prop_train=.6):
    
    ind_to_stop_train = int(prop_train*ts.shape[0])
    x_chunks, y_chunks, lab_tr = [], [], []
    x_val, y_val, lab_val = [], [], []
    start, stop = 0, ts.shape[0] - length - future_step + 1 
    for ind in range(start, stop) :
        if ind <= ind_to_stop_train - length - future_step + 1  :
            x_chunks.append(ts[ind:ind + length])
            y_chunks.append(ts[ind + length:ind + length+future_step][:, 0])
            lab_tr.append(label[ind + length:ind + length+future_step][:, 0])
        else :
            x_val.append(ts[ind:ind + length])
            y_val.append(ts[ind + length:ind + length+future_step][:, 0])
            lab_val.append(label[ind + length:ind + length+future_step][:, 0])

    to_return = np.asarray(x_chunks), np.asarray(y_chunks), np.asarray(x_val),\
                np.asarray(y_val), np.asarray(lab_tr), np.asarray(lab_val)
    return to_return


def generate_train_val(names, previous_step, dim, 
                       future_step, prop_train):
    X_train, X_val = np.empty((0, previous_step, dim)), np.empty((0, previous_step, dim))
    Y_train, Y_val = np.empty((0, future_step)), np.empty((0, future_step))
    label_train, label_val  = np.empty((0, future_step)), np.empty((0, future_step))
    
    for file_name in names :
        X, label = read_dataset(file_name, _normalize=True)
        x_chunks, y_chunks, x_val, y_val, lab_tr, lab_val = partition_ts(X, label, previous_step,
                                                                         future_step, prop_train)
        X_train = np.vstack((X_train, x_chunks))
        Y_train = np.vstack((Y_train, y_chunks))
        X_val = np.vstack((X_val, x_val))
        Y_val= np.vstack((Y_val, y_val))
        label_train = np.vstack((label_train, lab_tr))
        label_val = np.vstack((label_val, lab_val))
    return X_train, X_val, Y_train, Y_val, label_train, label_val

def plot_prediction(file_name, model, previous_step,
                    future_step, prop_train, name_model) :
    X, label = read_dataset(file_name, _normalize=True)
    _, _, x_val, y_val, _, label_val = partition_ts(X, label, previous_step,
                                                    future_step, prop_train)
    y_pred = model.predict(x_val)
    scores = np.linalg.norm(y_pred - y_val, axis=1)
    aucPerformance(scores, label_val)
    
    end = X.shape[0]
    start = end - y_pred.shape[0]
    fig, ax = plot_ts(file_name, start, end, zoom=True, _normalize=True)
    ax[1].plot(np.arange(start+1, end+1), y_val[:, 0], label="real_value", c="red")
    ax[1].plot(np.arange(start+1, end+1), y_pred[:, 0], label="predicted", c="blue")
    ax[1].legend(loc="best")
    fig.savefig(name_model + '.png')
    

names = glob.glob("data/A1Benchmark/*.csv")
fig, ax = plot_ts(names[3])


prop_train = .5
previous_step, dim, future_step = 32, 1, 1
X_train, X_val, Y_train, Y_val, label_train, label_val = generate_train_val(names, previous_step, dim, 
                                                                            future_step, prop_train)


model = TSPredictor(network_type="cnn", epochs=20,
                    batch_size=128, kernel_size=3, pool_size=2)
model.fit(X_train, Y_train, X_val, Y_val)

plot_prediction(names[3], model, previous_step,
                    future_step, prop_train, name_model="cnn_1")

scores = np.linalg.norm(model.predict(X_val) - Y_val, axis=1)

auc = aucPerformance(scores, label_val)









