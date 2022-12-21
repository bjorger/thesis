from helper.TestData import TestDataSingle, TestDataStacked
from typing import List
import pandas as pd
from helper.DataScaler import DataScaler

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

def create_test_cases(neurons, dropout_rate, interval, iteration):
    lrcPriceDataScaler = DataScaler(dataset=lrc_price_data_numeric, name='Loopring Price Snapshots', interval=interval)
    btcPriceDataScaler = DataScaler(dataset=btc_price_data_numeric, name='Bitcoin Price Snapshots', interval=interval)
    redditSentimentDataScaler = DataScaler(dataset=reddit_sentiment_amount_comments_numeric, name='Reddit Sentiment', interval=interval)
    tweetSentimentDataScaler = DataScaler(dataset=tweets_sentiment_followers_numeric, name='Twitter Sentiment', interval=interval)
    fearAndGreedDataScaler = DataScaler(dataset=fear_and_greed_numeric, name='Fear and Greed Index', interval=interval)
    
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
    
    testCasesBidirectional: List[TestDataStacked] = [
        TestDataStacked(
            name="Price_Twitter", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True), 
            TestDataStacked(
            name="Price_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),     
        TestDataStacked(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),       
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),
        TestDataStacked(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration,
            bidirectional=True),  
    ]
    
    testCasesSingleLayer: List[TestDataSingle] = [
        TestDataSingle(name='Price', neurons=128, dropout_rate=dropout_rate, inputs=[lrcPriceDataScaler], interval=interval),
        TestDataSingle(name='Price_Twitter', neurons=128, dropout_rate=0, inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], interval=interval),
    ]
    
    neurons = 128
    
    testCasesSingleLayer: List[TestDataSingle] = [
        TestDataSingle(
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
        TestDataSingle(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration),     
        TestDataSingle(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            interval=interval,
            iteration=iteration),       
        TestDataSingle(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
        TestDataSingle(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),
        TestDataSingle(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            interval=interval,
            iteration=iteration), 
        TestDataSingle(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration), 
        TestDataSingle(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            interval=interval,
            iteration=iteration),  
    ]
    
    return testCasesSingleLayer, testCasesMultiLayer, testCasesBidirectional