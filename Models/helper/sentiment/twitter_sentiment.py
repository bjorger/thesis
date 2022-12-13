import re
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import timedelta, datetime
"""
This updated version of the script uses a more elaborate approach to adjusting the sentiment score based on the number of likes and followers. In this approach, the sentiment score is adjusted based on the number of likes and followers, but with a maximum adjustment of 10 (i.e. a tweet with 10,000 likes or 10,000 followers will not receive any further adjustment to the sentiment score). This helps to prevent extremely large values of likes or followers from having an overly large influence on the sentiment score.

Again, this is just an example and the specific approach to adjusting the sentiment score may need to be fine-tuned for your specific use case.
"""
def calculate_tweet_sentiment_with_likes_and_follower(tweet, likes, followers):
    # Use VADER to calculate the sentiment of the tweet
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(tweet)

    # Calculate the sentiment score
    sentiment_score = sentiment['compound']

    # Adjust the sentiment score based on the number of likes and followers
    weighted_sentiment = sentiment_score * (likes + 1) * (followers + 1)
    return weighted_sentiment

def calculate_tweet_sentiment_with_likes(tweet, likes):
    # Use VADER to calculate the sentiment of the tweet
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(tweet)

    # Calculate the sentiment score
    sentiment_score = sentiment['compound']

    # Adjust the sentiment score based on the number of likes
    if likes > 0:
        sentiment_score *= (1 + min(likes, 10000) / 1000)

    # Keep the sentiment score in the range of -1 to 1
    #sentiment_score = min(1, max(-1, sentiment_score))

    return sentiment_score

def calculate_sentiment(tweet): 
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(tweet)

    # Calculate the sentiment score
    sentiment_score = sentiment['compound']
    
    return sentiment_score

months = {
    'January': 31,
    'February': 28,
    'March': 31,
    'April': 30,
    'May': 31,
    'June': 30,
    'July': 31,
    'August':31,
    'September':30,
    'October': 31,
    'November': 30,
    'December': 31,
}

def group_tweets_by_timewindow(tweets: pd.DataFrame, filename: str) -> pd.DataFrame:
    tweets['date'] = pd.to_datetime(tweets['date'])
    tweets.set_index('date')
    tweets.index = tweets['date']
    startDate = pd.to_datetime('2021-10-31 00:00:00')

    tweets = tweets[tweets.index >= startDate]
    
    analyzer = SentimentIntensityAnalyzer()

    analyzed_tweets = pd.DataFrame(columns=['Timewindow', 'Sentiment', 'Volume'])

    likeWeight = 0.2
    followerWeight = 0.4

    for year in range(2021, datetime.now().year + 1):
        tweets_year = tweets[tweets['date'].dt.year == year]
        for month in months:
            tweets_month = tweets_year[pd.to_datetime(tweets_year['date']).dt.strftime('%B') == month]
            for day in range(1, months[month] + 1):
                tweets_day = tweets_month.loc[(tweets_month['date'].dt.day == day)]

                if (tweets_day.size == 0):
                    continue
                
                for i in range(1, 49):
                    start = str(timedelta(minutes=(i-1)*30))
                    end = str(timedelta(minutes=(i*30)-1, seconds=59))
                    current_tweets = tweets_day.between_time(start, end)
                    results_tweet = []
                    results_tweet_likes = []
                    results_tweet_likes_followers = []
                    results_tweet_followers = []
                    tweets_in_timeframe = []
                    followers = 0
                    for _, tweet in current_tweets.iterrows():
                        tweet_id = str(tweet['id'])
                        if (tweet_id in tweets_in_timeframe):
                            continue
                        sentiment_result_followers_likes = calculate_tweet_sentiment_with_likes_and_follower(tweet['content'], int(tweet['likeCount']), tweet['followerCount'][0])
                        sentiment_result_likes = calculate_tweet_sentiment_with_likes(tweet['content'], int(tweet['likeCount']))
                        sentiment_result = calculate_sentiment(tweet['content'])
                        sentiment_result_followers = calculate_tweet_sentiment_with_likes(tweet['content'], tweet['followerCount'][0])
                        
                        results_tweet.append(sentiment_result)
                        results_tweet_likes.append(sentiment_result_likes)
                        results_tweet_likes_followers.append(sentiment_result_followers_likes)
                        results_tweet_followers.append(sentiment_result_followers)
                        
                        tweets_in_timeframe.append(tweet_id)
                        followers += tweet['followerCount'][0]
                    
                    sentiment = 0
                    sentiment_likes = 0
                    sentiment_followers = 0
                    sentiment_followers_likes = 0
                    timewindow = '{}-{}-{} {}-{}'.format(day, month, year, start, end)

                    if (len(results_tweet) > 0):
                        sentiment = sum(results_tweet) / len(results_tweet)
                        sentiment_likes = sum(results_tweet_likes) / len(results_tweet)
                        sentiment_followers = sum(results_tweet_followers) / len(results_tweet)
                        sentiment_followers_likes = sum(results_tweet_likes_followers) / len(results_tweet)
                        
                    entry = {
                        'Timewindow': timewindow,
                        'Sentiment': sentiment,
                        'Sentiment_Likes': sentiment_likes,
                        'Sentiment_Followers': sentiment_followers,
                        'Sentiment_Followers_Likes': sentiment_followers_likes,
                        'Volume': len(current_tweets),
                        'TweetIds': ','.join(tweets_in_timeframe),
                        'Followers': followers
                    }
                    analyzed_tweets = analyzed_tweets.append(entry, ignore_index=True)
                    
    analyzed_tweets.to_csv('data/test_data/{}.csv'.format(filename), sep=',')
    analyzed_tweets.to_json('data/test_data/{}.json'.format(filename), orient="records")
        



"""
tweets = pd.read_json('data/lrc_1year_unfiltered_tweets.json', orient="records")
users = pd.read_json('data/lrc_1year_unfiltered_users.json', orient="records")

followerCount = []
for _, tweet in tweets.iterrows():
    user = users.loc[users['username'] == tweet['user']]
    followerCount.append(user['followersCount'])
    
tweets_prepped = pd.DataFrame().assign(date=tweets['date'], id=str(tweets['id']), content=tweets['normalized'], followerCount=followerCount, likeCount=tweets['likeCount'])

tweets_prepped.to_csv('data/lrc_1year_tweets_with_followers.csv', sep='\t')
tweets_prepped.to_json('data/lrc_1year_tweets_with_followers.json', orient="records")
"""

tweets = pd.read_json('data/lrc_1year_tweets_with_followers.json', orient="records")

group_tweets_by_timewindow(tweets, "lrc_1year_sentiment_twitter")