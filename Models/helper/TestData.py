from typing import List
from helper.DataScaler import DataScaler
import os
import pandas as pd
class TestData():
    name: str
    filename: str
    dropout_rate: float
    inputs: List[DataScaler]
    interval: int
    iteration: int
    
    def __init__(self, name, dropout_rate, inputs: List[DataScaler], interval: int, iteration: int):
        self.name = name
        self.interval = interval
        self.dropout_rate = dropout_rate
        self.inputs = inputs
        self.iteration = iteration
        
    def generateResultString(self) -> str:
        pass
    
    
    def saveResults(self, rmse: float, mse: float, dropout_rate: float) -> None:
        pass
    
class TestDataStacked(TestData):
    neurons: List[int]
    columns = ['name', 'features', 'neurons', 'layers', 'dropout_rate', 'interval', 'rmse', 'mse']
    
    def __init__(self, name, dropout_rate, neurons, inputs: List[DataScaler], interval: int, iteration: int):
        super().__init__(name, dropout_rate, inputs, interval, iteration)
        self.neurons = neurons
        self.layers = len(neurons)
        self.filename = 'Stacked_{}_{}_{}_{}_{}_{}'.format(name, '_'.join(map(str, neurons)), dropout_rate, self.layers, interval, self.iteration)
        
    def generateResultString(self) -> str:
        return 'Name: {}\nNeurons: {}\nDropout Rate: {}\nLayers: {}'.format(self.name, ', '.join(map(str, self.neurons)), self.dropout_rate, self.layers)

    def saveResults(self, rmse: float, mse: float) -> None:
        result = pd.DataFrame(columns=self.columns)

        layers = len(self.neurons)
        feature_list = ''
        neurons = ''

        for i in range(0, len(self.inputs)):
            append = ', '
            if i == len(self.inputs) - 1:
                append = ''
            feature_list += (self.inputs[i].name + append)
        
        for i in range(0, len(self.neurons)):
            append = ', '
            if i == len(self.neurons) - 1:
                append = ''
            neurons += (str(self.neurons[i]) + append)
            
        entry = {
            'name': self.name,
            'features': feature_list,
            'neurons': neurons,
            'layers': layers,
            'dropout_rate': self.dropout_rate,
            'interval': self.interval,
            'rmse': rmse,
            'mse': mse
        }
        
        result = result.append(entry, ignore_index=True)

        if not os.path.isfile('results/LSTM/results_stacked.csv'):
            result.to_csv('results/LSTM/results_stacked.csv', columns=self.columns)
        else:
            result.to_csv('results/LSTM/results_stacked.csv', mode='a', header=False)
        
class TestDataSingle(TestData):
    neurons: int
    columns = ['name', 'features', 'neurons', 'dropout_rate', 'interval', 'rmse', 'mse']

    def __init__(self, name, dropout_rate, neurons, inputs: List[DataScaler], interval: int):
        super().__init__(name, dropout_rate, inputs, interval)
        self.neurons = neurons
        self.filename = 'Single_{}_{}_{}_{}'.format(name, neurons, dropout_rate, interval)

        
    def generateResultString(self) -> str:
        return 'Name: {}\nNeurons: {}\nDropout Rate: {}'.format(self.name, self.neurons, self.dropout_rate)
    
    # add interval & patience
    def saveResults(self, rmse: float, mse: float) -> None:
        result = pd.DataFrame(columns=self.columns)
        feature_list = ''
        neurons = ''

        for i in range(0, len(self.inputs)):
            append = ', '
            if i == len(self.inputs) - 1:
                append = ''
            feature_list += (self.inputs[i].name + append)

        entry = {
            'name': self.name,
            'features': feature_list,
            'neurons': str(neurons),
            'dropout_rate': self.dropout_rate,
            'interval': self.interval,
            'rmse': rmse,
            'mse': mse
        }
        
        result = result.append(entry, ignore_index=True)

        if not os.path.isfile('results/LSTM/results_single.csv'):
            result.to_csv('results/LSTM/results_single.csv', columns=self.columns)
        else:
            result.to_csv('results/LSTM/results_single.csv', mode='a', header=False)