import pandas as pd
from helper.DataScaler import DataScaler
from helper.LSTM_Model import LSTM
from typing import List
from helper.TestData import TestDataSingle, TestDataStacked
from Testcases import create_test_cases_single, create_test_cases_stacked

batch_train = 78
batch_predict = 7

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
    
neurons = [[128, 64, 32], [128, 64], [64, 32]]
interval = 14
iterations = 20
dropout_rate = 0.05

for iteration in range(0, iterations):
    for i in range(0, len(neurons)):
        testCasesMultiLayer, testCasesBidirectional = create_test_cases_stacked(neurons[i], iteration)
        
        for i in range(0, len(testCasesMultiLayer)):
            testData = testCasesMultiLayer[i]      
            evaluate_stacked_layer_model(testData)
    
        for i in range(0, len(testCasesBidirectional)):
            testData = testCasesBidirectional[i]      
            evaluate_stacked_layer_model(testData)
            
    testCasesSingleLayer = create_test_cases_single(iteration)
    
    for i in range(0, len(testCasesSingleLayer)):
        testData = testCasesSingleLayer[i]
        evaluate_single_layer_model(testData)
        
    print("\n\n\n-----------------------------------------------------------")
    print('Successfully completed iteration: {}'.format(iteration))
    print("-----------------------------------------------------------\n\n\n")