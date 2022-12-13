import pandas as pd
import matplotlib.pyplot as plt
import datetime
"""
tweets = pd.read_json('data/lrc_1year_unfiltered_tweets.json', orient="records")

print(len(tweets.keys()))
"""
"""
tweets = pd.read_json('data/tweets.json', orient="records")

print(len(tweets.index))
"""
"""
lrc_display = pd.read_json('data/lrc_display_tweets.json', orient="records")
lrc_users = pd.read_json('data/lrc_display_users.json', orient="records")
print(len(lrc_display.keys()))
print(lrc_display['retweetedTweet'])

print(lrc_users)
print(lrc_users['followersCount'])
"""
"""
print(lrc_users.keys())
"""
lrc_price = pd.read_csv('data/lrc_coin_api.csv', sep=",")
plt.figure(figsize=(16,8))
plt.title('LRC prices')
plt.xlabel('Date')
plt.ylabel('Opening Price USD ($)')
plt.plot(lrc_price['rate_open'])
plt.legend(['Val'], loc='lower right')
plt.show()