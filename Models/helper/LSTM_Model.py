from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from typing import List
import numpy as np
import pandas as pd
from helper.DataScaler import DataScaler
from keras import callbacks
from keras.layers import concatenate
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from helper.TestData import TestData

earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 3, 
                                        restore_best_weights = True)

class LSTM():
    
    def __init__(self, inputs: List[DataScaler], scaler: MinMaxScaler,name: str) -> None:
        self.x_train = []
        self.x_test = []
        
        self.scaler = scaler
        self.inputs = inputs
        self.y_train = inputs[0].y_train
        self.y_test = inputs[0].y_test
        self.name = name
        for input in inputs:
            self.x_train.append(input.x_train)
            self.x_test.append(input.x_test)
            
    def create_model(self, batch_size, neurons, dropout_rate):
        inputs = []
        for scaler in self.inputs:
            batch_shape=(batch_size, scaler.x_train.shape[1], scaler.x_train.shape[2])
            inputs.append(keras.layers.Input(batch_shape=batch_shape))

        input = concatenate(inputs)
        lstm_layer1 = keras.layers.LSTM(neurons, return_sequences=False, stateful=True)(input)
        dropout_layer1 = keras.layers.Dropout(dropout_rate)(lstm_layer1)
        output = keras.layers.Dense(1)(dropout_layer1)
        
        self.model = keras.models.Model(inputs=inputs, outputs=output)
        self.model.summary()
        
    def create_layered_model(self, batch_size, dropout_rate: float, neurons: List[int]):
        inputs = []
        for scaler in self.inputs:
            inputs.append(keras.layers.Input(batch_shape=(batch_size, scaler.x_train.shape[1], scaler.x_train.shape[2])))           

        input = concatenate(inputs)
        layers: List[keras.layers] = []

        for i in range(0, len(neurons)):
            if i == 0:
                """
                First Layer
                """
                lstm_layer = keras.layers.LSTM(neurons[i], return_sequences=True, stateful=True)(input)
            elif i == len(neurons) - 1:
                """
                Last Layer
                """
                lstm_layer = keras.layers.LSTM(neurons[i], return_sequences=False, stateful=True)(layers[i-1])
                layers.append(lstm_layer)
                break
            else:
                """
                Layers in between
                """
                lstm_layer = keras.layers.LSTM(neurons[i], return_sequences=True, stateful=True)(layers[i-1])
            dropout_layer = keras.layers.Dropout(dropout_rate)(lstm_layer)
            layers.append(dropout_layer)
        
        output = keras.layers.Dense(1)(layers.pop())
        self.model = keras.models.Model(inputs=inputs, outputs=output)
        self.model.summary()
                        
    def train_model(self, batch_size):
        # Look up best metrics for LSTM
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=100, callbacks=[earlystopping])
        self.model.save_weights('results/LSTM/trained_models/lstm_{}.h5'.format(self.name))
        
    def predict(self, batch_size):
        self.model.load_weights('results/LSTM/trained_models/lstm_{}.h5'.format(self.name))
        predictions = self.model.predict(self.x_test, batch_size=batch_size)

        self.predictions = self.scaler.inverse_transform(predictions)
        self.predictions = self.predictions.flatten()
            
        self.calculateMetrics()
    
    def showPlot(self): 
        validation = pd.DataFrame()
        validation['price'] = self.y_test
        validation['Predictions'] = self.predictions

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.plot(validation[['price', 'Predictions']])
        plt.legend(['Val', 'Predictions'], loc='lower right')
        plt.savefig('results/LSTM/plots/{}.png'.format(self.name))
        
    def calculateMetrics(self):
        predictions = self.predictions
        self.mse = np.mean(np.abs(self.y_test - predictions))
        self.rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
    