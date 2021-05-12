from get_PF_Features import get_PF_Features
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn,optim
from torch.autograd import Variable
from  torch.nn import init
import seaborn as sns; sns.set()
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *

lag = 1
class lstmIGT(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,layer_num):
        super().__init__()
        self.lstmLayer=nn.LSTM(in_dim,hidden_dim,layer_num)
        self.relu=nn.ReLU()
        self.fcLayer=nn.Linear(hidden_dim,out_dim)
        self.weightInit = (np.sqrt(1.0/hidden_dim))
    def forward(self, x):
        out,_=self.lstmLayer(x)
        out=self.relu(out)
        out=self.fcLayer(out)
        out = nn.Softmax(dim=-1)(out)
        return out

def getTS(r):
    ts = np.zeros((r.shape[0],r.shape[1],4))
    for i in np.arange(4):
        ts[r==i+1,i] = 1
    return ts

def lstm_pred(train_data, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr):
    Feats = get_PF_Features(Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr)
    n_runs=5
    prediction = np.repeat([0], 5)
    n_nodes, n_layers = 10, 2
    prediction.shape = (1, 5)
    for run in range(n_runs):
        # train a random forest algorithm using all supplied features of the train data

        x_train = train_data.iloc[:, 1:38]
        y_train = train_data['B_rate']
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        print(x_train.shape)
        model = Sequential()
        #Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        # Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        # Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        # Adding the output layer
        model.add(Dense(units = 1))
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#         rf_model = RandomForestRegressor(n_estimators=500, max_features=0.3333, min_samples_leaf=5)
        model.fit(x_train, y_train, epochs = 10, batch_size = 32)
        # let the trained RF predict the prediction prbolem
        Feats = np.array(Feats)
        Feats = np.reshape(Feats, (Feats.shape[0], Feats.shape[1], 1))
        pred = model.predict(Feats)
        print(pred.shape)
        prediction = np.add(prediction, (1 / n_runs) * pred)

    return prediction
