from typing import List
from helper.DataScaler import DataScaler

class TestData():
    name: str
    filename: str
    dropout_rate: float
    batch_train = 64
    batch_predict = 1
    interval = 24
    inputs: List[DataScaler]
    
    def __init__(self, name, dropout_rate, inputs: List[DataScaler]):
        self.name = name
        self.dropout_rate = dropout_rate
        self.inputs = inputs
        
    def generateResultString(self) -> str:
        pass
    
class TestDataStacked(TestData):
    neurons: List[int]
    
    def __init__(self, name, dropout_rate, neurons, inputs: List[DataScaler]):
        super().__init__(name, dropout_rate, inputs)
        self.neurons = neurons
        self.filename = '{}_{}_{}'.format(name, '_'.join(neurons), dropout_rate)
        
    def generateResultString(self) -> str:
        return 'Name: {}\nNeurons: {}\nDropout Rate: {}'.format(self.name, ', '.join(self.neurons), self.dropout_rate)

        
        
class TestDataSingle(TestData):
    neurons: int
    
    def __init__(self, name, dropout_rate, neurons, inputs: List[DataScaler]):
        super().__init__(name, dropout_rate, inputs)
        self.neurons = neurons
        self.filename = '{}_{}_{}'.format(name, neurons, dropout_rate)

        
    def generateResultString(self) -> str:
        return 'Name: {}\nNeurons: {}\nDropout Rate: {}'.format(self.name, self.neurons, self.dropout_rate)