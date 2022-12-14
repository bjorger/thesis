import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from helper.DataScaler import DataScaler
from helper.LSTM_Model import LSTM
from typing import List
from helper.TestData import TestDataSingle, TestDataStacked

"""
Data
"""
pricedata = pd.read_csv('helper/sentiment/data/test_data/lrc_1year_price_snapshot.csv', sep=',')
tweets = pd.read_csv('helper/sentiment/data/test_data/lrc_1year_sentiment_twitter.csv', sep=',')

price_data_numeric = pd.to_numeric(pricedata['rate_open'])
priceDataScaler = DataScaler(dataset=price_data_numeric, name='Price')
batch_train = 64
batch_predict = 1


testCasesSingleLayer: List[TestDataSingle] = [
    TestDataSingle(name='Price_Price', neurons=126, dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataSingle(name='Price_Price', neurons=256, dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataSingle(name='Price_Price', neurons=512, dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataSingle(name='Price_Price', neurons=126, dropout_rate=0.3, inputs=[priceDataScaler]),
    TestDataSingle(name='Price_Price', neurons=256, dropout_rate=0.3, inputs=[priceDataScaler]),
    TestDataSingle(name='Price_Price', neurons=512, dropout_rate=0.3, inputs=[priceDataScaler]),
]

testCasesMultiLayer: List[TestDataStacked] = [
    TestDataStacked(name="Price_Price", neurons=[126, 126, 126], dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataStacked(name="Price_Price", neurons=[126, 256, 512], dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataStacked(name="Price_Price", neurons=[256, 256, 256], dropout_rate=0.2, inputs=[priceDataScaler]),
    TestDataStacked(name="Price_Price", neurons=[126, 126, 126], dropout_rate=0.3, inputs=[priceDataScaler]),
    TestDataStacked(name="Price_Price", neurons=[126, 256, 512], dropout_rate=0.3, inputs=[priceDataScaler]),
    TestDataStacked(name="Price_Price", neurons=[256, 256, 256], dropout_rate=0.3, inputs=[priceDataScaler]),
]


def evaluate_single_layer_model(testData: TestDataSingle):
    print("\n\n\n-----------------------------------------------------------")
    print("Evaluating Model: {}".format(testData.filename))
    print("-----------------------------------------------------------\n\n\n")

    model = LSTM(inputs=testData.inputs, scaler=testData.inputs[0].scaler, name=testData.filename)

    #https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
    model.create_model(batch_size=testData.batch_train, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
    model.train_model(batch_size=testData.batch_train)
    model.create_model(batch_size=testData.batch_predict, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
    model.predict(batch_size=testData.batch_predict)
    model.showPlot()
    model.printResult(input_data=testData)
    
    
def evaluate_stacked_layer_model(testData: TestDataStacked):
    print("\n\n\n-----------------------------------------------------------")
    print("Evaluating Model: {}".format(testData.filename))
    print("-----------------------------------------------------------\n\n\n")

    model = LSTM(inputs=testData.inputs, scaler=testData.inputs[0].scaler, name=testData.filename)

    #https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
    model.create_layered_model(batch_size=testData.batch_train, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
    model.train_model(batch_size=testData.batch_train)
    model.create_layered_model(batch_size=testData.batch_predict, neurons=testData.neurons, dropout_rate=testData.dropout_rate)
    model.predict(batch_size=testData.batch_predict)
    model.showPlot()
    model.printResult(input_data=testData)
    

for i in range(0, len(testCasesSingleLayer)):
    evaluate_single_layer_model(testCasesSingleLayer[i])


for i in range(0, len(testCasesMultiLayer)):
    evaluate_stacked_layer_model(testCasesMultiLayer[i])