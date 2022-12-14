
import pandas as pd
#from Analzyer import group_tweets_by_timewindow
import requests
from pandas.io.json import json_normalize
import praw

"""
tweets = pd.read_csv('data/lrc_1year_analzyed.csv', sep=',')

print(tweets['Volume'].sum())
"""
"""
tweets = pd.read_json('data/lrc_1year_unfiltered_tweets.json', orient="records")
group_tweets_by_timewindow(tweets, 'lrc_1year_unfiltered')
"""
"""
prices = pd.read_csv('data/lrc_1year_coin_api.csv', sep=',')
print(len(prices.index))
prices = prices.drop_duplicates('rate_open')
print(len(prices.index))
"""
"""
prices.to_csv('data/lrc_1year_coin_api_cleaned.csv', sep=',')
prices = pd.read_csv('data/lrc_1year_unfiltered_analzyed.csv', sep=',')
print(len(prices.index))
prices = prices.drop_duplicates('Timewindow')
print(len(prices.index))
prices.to_csv('data/lrc_1year_unfiltered_analyzed_cleaned.csv', sep=',')
"""
"""
tweets = pd.read_json('data/lrc_1year_unfiltered_tweets.json', orient="records")
print(len(tweets.index))
"""
"""
reddit = pd.read_json('data/reddit_lrc_1year.json', orient="records")
print(len(reddit.index))
""" 
"""
fng = "https://api.alternative.me/fng/?limit=0"
response = requests.get(fng)
res_dict = response.json()
res_dict = json_normalize(res_dict['data'])
res_df = pd.DataFrame(res_dict)

print(len(res_df.index))
"""
"""
config = {
    "username" : "bjorgbirb",
    "client_id" : "-QdhQYEx5aOdyl-p-UfoVQ",
    "client_secret" : "Zu1RswYSAKjFVABFb7AUTsKuAhg7IQ",
    "user_agent" : "script:https://github.com/bjorger/master-thesis:v1.0.0 (by u/bjorgbirb)"
}

praw_reddit = praw.Reddit(client_id = config['client_id'], \
                     client_secret = config['client_secret'], \
                     user_agent = config['user_agent'])


reddit = pd.read_json('sentiment/data/final/reddit_lrc_1year.json', orient="records")
print(len(reddit.index))
print(reddit.head())

print(len(reddit.keys()))
for k in reddit.keys():
    print(k)
print(reddit['selftext'])
"""
"""
authors = reddit['author']
authors.dropna(inplace=True)
authors.drop_duplicates(inplace=True)
#print(len(authors.index))

for author in authors:
    user = praw_reddit.redditor(author)
    print(user.comment_karma)
    break

#df = pd.DataFrame(authors)
"""
