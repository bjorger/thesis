import pandas as pd
from helper.DataScaler import DataScaler
from helper.LSTM_Model import LSTM
from typing import List
from helper.TestData import TestDataSingle, TestDataStacked
from Testcases import create_test_cases

def train_lstm(
    batch_train: int, 
    batch_predict: int, 
    interval: int, 
    iteration: int, 
    neurons: List[int], 
    dropout_rate: float) -> None:


    def evaluate_single_layer_model(testData: TestDataSingle):
        print("\n\n\n-----------------------------------------------------------")
        print("Evaluating Model: {}".format(testData.filename))
        print("-----------------------------------------------------------\n\n\n")

        model = LSTM(inputs=testData.inputs, scaler=testData.inputs[0].scaler, name=testData.filename)

        #https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
        model.create_model(batch_size=batch_train, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
        model.train_model(batch_size=batch_train)
        model.create_model(batch_size=batch_predict, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
        model.predict(batch_size=batch_predict)
        model.showPlot()
        testData.saveResults(model.rmse, model.mse)

        
    def evaluate_stacked_layer_model(testData: TestDataStacked):
        print("\n\n\n-----------------------------------------------------------")
        print("Evaluating Model: {}".format(testData.filename))
        print("-----------------------------------------------------------\n\n\n")

        model = LSTM(inputs=testData.inputs, scaler=testData.inputs[0].scaler, name=testData.filename)

        #https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
        model.create_layered_model(batch_size=batch_train, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
        model.train_model(batch_size=batch_train)
        model.create_layered_model(batch_size=batch_predict, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
        model.predict(batch_size=batch_predict)
        model.showPlot()
        testData.saveResults(model.rmse, model.mse)
        
     
    testCasesSingleLayer, testCasesMultiLayer, testCasesBidirectional = create_test_cases(
        dropout_rate=dropout_rate, 
        interval=interval, 
        iteration=iteration, 
        neurons=neurons)
    
    for i in range(0, len(testCasesSingleLayer)):
        testData = testCasesSingleLayer[i]
        evaluate_single_layer_model(testData)
        
    for i in range(0, len(testCasesMultiLayer)):
        testData = testCasesMultiLayer[i]      
        evaluate_stacked_layer_model(testData)
    
    for i in range(0, len(testCasesBidirectional)):
        testData = testCasesBidirectional[i]      
        evaluate_stacked_layer_model(testData)
        


neurons = [[128, 64, 32], [256, 128], [128, 64]]
batch_size_train = 78
batch_size_predict = 7
interval = 14
iterations = 20
dropout_rate = 0.05

for iteration in range(0, iterations):
    for i in range(0, len(neurons)):
        train_lstm(
            batch_size_train, 
            batch_size_predict, 
            interval, 
            iteration, 
            neurons[i], 
            dropout_rate
        )
        print("\n\n\n-----------------------------------------------------------")
        print('Successfully completed iteration: {}'.format(iteration))
        print("-----------------------------------------------------------\n\n\n")

"""
find best model (single or layered)
A layered model is better for multiple inputs:
Best RMSE for single layer: 0.003956195987610552
Best RMSE for stacked layer: 0.002961009874215429

For a single input it is a single layer with the following:
0,Price,Loopring Price Snapshots,,0,7,0.0022391782733306212,0.0016226423837650415

best interval: 14
best dropout: dropout_rate
"""
