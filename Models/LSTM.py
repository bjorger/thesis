import yfinance as yf
import pandas as pd
from helper.DataScaler import DataScaler
from helper.LSTM_Model import LSTM
from typing import List
from helper.TestData import TestDataSingle, TestDataStacked
from itertools import chain, combinations, product
from pprint import pprint
"""
Data
"""
lrc_pricedata = pd.read_csv('test_data/lrc_1year_price_snapshot.csv', sep=',')
btc_pricedata = pd.read_csv('test_data/btc_1year_coin_api.csv', sep=',')
tweets = pd.read_csv('test_data/lrc_1year_sentiment_twitter.csv', sep=',')
reddit = pd.read_csv('test_data/lrc_1year_sentiment_reddit.csv', sep=',')
fear_and_greed = pd.read_csv('test_data/fng_prepped.csv', sep=',')

lrc_price_data_numeric = pd.to_numeric(lrc_pricedata['rate_open'])
btc_price_data_numeric = pd.to_numeric(btc_pricedata['rate_open'])
reddit_sentiment_numeric = pd.to_numeric(reddit['Sentiment'])
reddit_sentiment_amount_comments_numeric = pd.to_numeric(reddit['Sentiment_amount_comments'])
reddit_sentiment_likes_numeric = pd.to_numeric(reddit['Sentiment_Likes'])
tweets_sentiment_numeric = pd.to_numeric(tweets['Sentiment'])
tweets_sentiment_followers_numeric = pd.to_numeric(tweets['Sentiment_Followers'])
tweets_sentiment_likes_numeric = pd.to_numeric(tweets['Sentiment_Likes'])
fear_and_greed_numeric = pd.to_numeric(fear_and_greed['Value'])

def train_lstm(
    batch_train: int, 
    batch_predict: int, 
    interval: int, 
    iteration: int, 
    neurons: List[int], 
    dropout_rate: float) -> None:
    
    lrcPriceDataScaler = DataScaler(dataset=lrc_price_data_numeric, name='Loopring Price Snapshots', interval=interval)
    btcPriceDataScaler = DataScaler(dataset=btc_price_data_numeric, name='Bitcoin Price Snapshots', interval=interval)
    redditSentimentDataScaler = DataScaler(dataset=reddit_sentiment_amount_comments_numeric, name='Reddit Sentiment', interval=interval)
    tweetSentimentDataScaler = DataScaler(dataset=tweets_sentiment_followers_numeric, name='Twitter Sentiment', interval=interval)
    fearAndGreedDataScaler = DataScaler(dataset=fear_and_greed_numeric, name='Fear and Greed Index', interval=interval)
    
    """
    testCasesSingleLayer: List[TestDataSingle] = [
        TestDataSingle(name='Price', neurons=128, dropout_rate=0, inputs=[lrcPriceDataScaler], interval=interval),
        TestDataSingle(name='Price_Twitter', neurons=128, dropout_rate=0, inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], interval=interval),
    ]

    """
        
    testCasesMultiLayer: List[TestDataStacked] = [
        TestDataStacked(
            name="Price_Twitter", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], 
            interval=interval,
            iteration=iteration), 
            TestDataStacked(
            name="Price_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration),     
        TestDataStacked(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),       
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataStacked(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),
        TestDataStacked(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration), 
        TestDataStacked(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration), 
        TestDataStacked(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        
    ]
    

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

    for i in range(0, len(testCasesMultiLayer)):
        testData = testCasesMultiLayer[i]      
        evaluate_stacked_layer_model(testData)

neurons = [[512, 256, 128], [1024, 512, 256], [1024, 512], [512, 256], [256, 128]]
batch_size_train = [78, 64, 61]
batch_size_predict = 7
intervals = [14, 24, 7]
iterations = 1
dropout_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for iteration in range(0, iterations):
    for z in range(0, len(intervals)):
        for i in range(0, len(neurons)):
            for j in range(0, len(dropout_rate)):
                train_lstm(
                    batch_size_train[z], 
                    batch_size_predict, 
                    intervals[z], 
                    iteration, 
                    neurons[i], 
                    dropout_rate[j]
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
