import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import torch
import numpy as np
def load_data(data_path,label_path,save_fig = False):
    # set dataset path
    train_data_path = data_path
    train_data_label_path = label_path
    # read data
    data_raw = pd.read_csv(train_data_path,parse_dates=["date"])
    label = pd.read_csv(train_data_label_path)

    # save histgram
    if save_fig:
        data_raw["avg_temp"].hist()
        plt.xlabel("日別平均気温(摂氏)")
        plt.ylabel("日数")
        plt.savefig("figure/train_data_hist.png")

    return data_raw,label

def norm_data(data_raw,target_key,save_fig = False):
    
    # normalized data 
    x_min, x_max = data_raw[target_key].min(), data_raw[target_key].max()
    data_raw[target_key] = (data_raw[target_key] - x_min) / ( x_max - x_min)

    # save histgram of normalized data
    if save_fig:
        data_raw[target_key].hist()
        plt.xlabel("日別平均気温(摂氏)")
        plt.ylabel("日数")
        plt.savefig("figure/train_data_norm_hist.png")

    return data_raw

def load_X_Y(data_raw, label):

    # # normalize data
    # data_raw = norm_data(data_raw,"avg_temp")

    # sperate data by year (each 43 days data into an array every year) 
    data_raw["year"]= data_raw["date"].dt.year # add the year column
    grouped_data = data_raw.groupby("year") # groupe data by year

    X = []
    for year, group in grouped_data:
        X.append(group["avg_temp"].values)

    # covert dataframe data to numpy array
    X = np.array(X)
    Y = np.array(label["day"].values)
    Y = np.reshape(Y,((len(Y)),1)) # reshape (1,n) to (n, 1) array

    print(f"X: {X.shape}, Y: {Y.shape}")
    return X, Y


