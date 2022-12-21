from typing import List
from helper.DataScaler import DataScaler
import os
import pandas as pd
class TestData():
    name: str
    filename: str
    dropout_rate: float
    inputs: List[DataScaler]
    iteration: int
    
    def __init__(self, name, dropout_rate, inputs: List[DataScaler],iteration: int):
        self.name = name
        self.dropout_rate = dropout_rate
        self.inputs = inputs
        self.iteration = iteration
        
    def saveResults(self, rmse: float, mse: float, dropout_rate: float) -> None:
        pass
    
class TestDataStacked(TestData):
    neurons: List[int]
    columns = ['name', 'features', 'neurons', 'layers', 'dropout_rate', 'rmse', 'mse', 'architecture']
    
    def __init__(self,
                name, 
                dropout_rate, 
                neurons, 
                inputs: List[DataScaler], 
                iteration: int,
                bidirectional: bool = False):
        super().__init__(name, dropout_rate, inputs, iteration)
        self.neurons = neurons
        self.layers = len(neurons)
        self.bidirectional = bidirectional
        self.prefix = 'bidirectional' if self.bidirectional else 'stacked'
        self.filename = '{}_{}_{}_{}_{}_{}'.format(
            self.prefix, name, 
            '_'.join(map(str, neurons)), 
            dropout_rate, 
            self.layers, 
            self.iteration)

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
            'rmse': rmse,
            'mse': mse,
            'architecture': self.prefix
        }
        
        result = result.append(entry, ignore_index=True)
        
        file = 'results_bidirectional' if self.bidirectional else 'results_stacked'
        path = 'results/LSTM/{}.csv'.format(file)

        if not os.path.isfile(path):
            result.to_csv(path, columns=self.columns)
        else:
            result.to_csv(path, mode='a', header=False)
        
class TestDataSingle(TestData):
    neurons: int
    columns = ['name', 'features', 'neurons', 'dropout_rate', 'rmse', 'mse']
    
    def __init__(
        self, 
        name, 
        dropout_rate, 
        neurons, 
        inputs: List[DataScaler], 
        iteration: int):
        super().__init__(name, dropout_rate, inputs, iteration)
        self.neurons = neurons
        self.filename = 'Single_{}_{}_{}_{}'.format(name, neurons, dropout_rate, iteration)

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
            'rmse': rmse,
            'mse': mse
        }
        
        result = result.append(entry, ignore_index=True)

        if not os.path.isfile('results/LSTM/results_single.csv'):
            result.to_csv('results/LSTM/results_single.csv', columns=self.columns)
        else:
            result.to_csv('results/LSTM/results_single.csv', mode='a', header=False)