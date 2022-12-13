import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from helper.DataScaler import DataScaler
from helper.LSTM_Model import LSTM
from helper.LSTM_old import LSTM_old

tweets = pd.read_csv('helper/sentiment/data/test_data/lrc_1year_sentiment_twitter.csv', sep=',')
pricedata = pd.read_csv('helper/sentiment/data/test_data/lrc_1year_price_snapshot.csv', sep=',')
price_data_numeric = pd.to_numeric(pricedata['rate_open'])
sentiment_data_numeric = pd.to_numeric(tweets['Sentiment_Followers_Likes'])

batch_train = 64
batch_predict = 1
interval = 24
epoch = 100

priceDataScaler = DataScaler(dataset=price_data_numeric, interval=interval, name='Price')
priceDataScaler.createTrainingData()
priceDataScaler.createTestData()

sentimentDataScaler = DataScaler(dataset=sentiment_data_numeric, interval=interval, name='Sentiment')
sentimentDataScaler.createTrainingData()
sentimentDataScaler.createTestData()

model = LSTM(
    inputs=[priceDataScaler, sentimentDataScaler], 
    scaler=priceDataScaler.scaler, 
    name='LSTM_price_sentiment_weighted_followers_likes_{}_{}_{}'.format(batch_train, batch_predict, epoch))

#https://stackoverflow.com/questions/43702481/why-does-keras-lstm-batch-size-used-for-prediction-have-to-be-the-same-as-fittin
model.create_model(batch_train)
model.train_model(100, batch_train)
model.create_model(batch_predict)
model.predict(batch_predict)
model.showPlot(batch_train, batch_predict)