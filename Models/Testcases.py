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

lrcPriceDataScaler = DataScaler(dataset=lrc_price_data_numeric, name='Loopring Price Snapshots')
btcPriceDataScaler = DataScaler(dataset=btc_price_data_numeric, name='Bitcoin Price Snapshots')
redditSentimentDataScaler = DataScaler(dataset=reddit_sentiment_amount_comments_numeric, name='Reddit Sentiment')
tweetSentimentDataScaler = DataScaler(dataset=tweets_sentiment_followers_numeric, name='Twitter Sentiment')
fearAndGreedDataScaler = DataScaler(dataset=fear_and_greed_numeric, name='Fear and Greed Index')

def create_test_cases_stacked(neurons, iteration):
    dropout_rate = 0.05    

    testCasesMultiLayer: List[TestDataStacked] = [
        TestDataStacked(
            name="Price", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler], 
            iteration=iteration), 
        TestDataStacked(
            name="Price_Twitter", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], 
            iteration=iteration), 
        TestDataStacked(
            name="Price_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            iteration=iteration),     
        TestDataStacked(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),       
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler,btcPriceDataScaler],      
            iteration=iteration),  
        TestDataStacked(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler],   
            iteration=iteration),
        TestDataStacked(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler],   
            iteration=iteration), 
        TestDataStacked(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration), 
        TestDataStacked(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
    ]
    
    testCasesBidirectional: List[TestDataStacked] = [
        TestDataStacked(
            name="Price", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_Twitter", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], 
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            iteration=iteration,
            bidirectional=True),     
        TestDataStacked(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration,
            bidirectional=True),       
        TestDataStacked(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),  
        TestDataStacked(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),
        TestDataStacked(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True), 
        TestDataStacked(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration,
            bidirectional=True),  
    ]
    
    return testCasesMultiLayer, testCasesBidirectional
    
def create_test_cases_single(iteration, neurons):
    dropout_rate = 0.05    
    
    testCasesSingleLayer: List[TestDataSingle] = [
        TestDataSingle(
            name="Price", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler], 
            iteration=iteration), 
        TestDataSingle(
            name="Price_Twitter", 
            neurons=neurons,
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler], 
            iteration=iteration), 
        TestDataSingle(
            name="Price_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            iteration=iteration),     
        TestDataSingle(
            name="Price_Twitter_Reddit", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Twitter_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler], 
            iteration=iteration),       
        TestDataSingle(
            name="Price_Twitter_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, tweetSentimentDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
        TestDataSingle(
            name="Price_Reddit_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration),
        TestDataSingle(
            name="Price_Reddit_FNG", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler], 
            iteration=iteration), 
        TestDataSingle(
            name="Price_Reddit_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, redditSentimentDataScaler, btcPriceDataScaler], 
            iteration=iteration), 
        TestDataSingle(
            name="Price_FNG_BTC", 
            neurons=neurons, 
            dropout_rate=dropout_rate, 
            inputs=[lrcPriceDataScaler, fearAndGreedDataScaler, btcPriceDataScaler], 
            iteration=iteration),  
    ]
    
    return testCasesSingleLayer