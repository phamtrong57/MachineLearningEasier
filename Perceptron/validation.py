from sakura_perceptron import SakuraPerceptron
from data_loader import load_X_Y,load_data,norm_data
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def validation(test_data_path, test_data_label_path):
    model = SakuraPerceptron()
    model.load_state_dict(torch.load("sakura_model_second.pt"))

    # set validation mode
    model.eval()

   
    data_raw, label  = load_data(test_data_path,test_data_label_path)
    data_raw = norm_data(data_raw,"avg_temp")
    X, Y = load_X_Y(data_raw,label)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    Y_predict_data = np.array([])
    year = 2015 
    with torch.no_grad():
        for X, Y in list(zip(X,Y)):
            outputs = model(X)
            loss = nn.MSELoss()(outputs,Y)
            # print(f"Year: {year}, Predict: {outputs.numpy()}, Truth: {Y.numpy()}")
            print(f"Year: {year}, Predict: {outputs.numpy().astype(int)}, Truth: {Y.numpy()}")
            year+=1
    np.savetxt("predict_data1.txt",Y_predict_data)

test_data_path = "Data/test_data.csv"
test_data_label_path = "Data/test_data_label.csv"

validation(test_data_path,test_data_label_path)