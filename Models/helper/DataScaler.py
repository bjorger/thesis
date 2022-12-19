import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from typing import List
from sklearn.model_selection import TimeSeriesSplit

class DataScaler():
    scaler: MinMaxScaler = None
    scaled_data = None
    train_data = None
    train_data_len = 0
    test_data_len = 0
    dataset: pd.Series = None
    x_train = []
    y_train = []
    x_test = None
    y_test = None
    interval = 24
    name = ''
    
    def __init__(self, dataset: pd.Series, name: str, interval = 24) -> None:
        self.name = name
        self.interval = interval
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.train_data_len = math.ceil(len(dataset.values)* 0.8)
        scaled_data = self.scaler.fit_transform(dataset.values.reshape(-1,1))
        self.dataset = dataset
                
        self.scaled_data = scaled_data

        self.train_data = scaled_data[0: self.train_data_len, :]
        
        self.createTrainingData()
        self.createTestData()

    def createTrainingData(self):
        x_train = []
        y_train = []
        for i in range(self.interval, self.train_data_len):
            x_train.append(self.train_data[i-self.interval:i, 0])
            y_train.append(self.train_data[i, 0])
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        self.x_train = x_train
        self.y_train = y_train
        
    def createTestData(self):
        test_data = self.scaled_data[self.train_data_len-self.interval: , : ]
        self.y_test = self.dataset[self.train_data_len:]
        x_test = []
        for i in range(self.interval, len(test_data)):
            x_test.append(test_data[i-self.interval:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        self.x_test = x_test