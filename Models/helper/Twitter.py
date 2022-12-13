import snscrape.modules.twitter as sntwitter
import pandas as pd
from TweetNormalizer import normalizeTweet
from Analzyer import group_tweets_by_timewindow

import re
from typing import Dict

def remove_from_dict(tweet: Dict, keys_to_remove: Dict) -> Dict:
    for key in keys_to_remove:
        del tweet[key]
    
    return tweet

def scrap_tweets(query: str, since: str, until: str, filename: str) -> pd.DataFrame:
    tweet_keys_to_remove = ['cashtags', 'coordinates', 'inReplyToTweetId', 'media', 'source', 'sourceLabel', 'tcooutlinks', 'url', 'outlinks', 'renderedContent', 'sourceUrl']
    user_keys_to_remove = ['location', 'linkUrl', 'linkTcourl', 'profileImageUrl', 'profileBannerUrl']
    
    tweets = []
    users = []
    users_stored = []
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:{} until:{}'.format(query, since, until)).get_items()):        
        tweet = tweet.__dict__
        user = tweet['user'].__dict__
                
        if tweet['lang'] != 'en':
            continue
        
        tweet = remove_from_dict(tweet, tweet_keys_to_remove)
        user = remove_from_dict(user, user_keys_to_remove)
        #Remove mentions
        tweet['content'] = re.sub("@[A-Za-z0-9_]+","", tweet['content'])
        #Remove hashtags
        tweet['content'] = re.sub("#[A-Za-z0-9_]+","", tweet['content'])
        normalized = normalizeTweet(tweet['content'])
        tweet['normalized'] = normalized
        #tweet['sentiment'] = analyzer.analyze_tweet(normalized)
        tweet['user'] = user['username']
        tweet['user_follower_count'] = user['followersCount']
        tweets.append(tweet)
        if user['id'] not in users_stored:
            users.append(user)  
            users_stored.append(user['id'])

    print('Successfully scrapped all tweets and users with {}'.format(query))
    tweets = pd.DataFrame(tweets) 
    tweets.reset_index(inplace=True)
    
    tweets.to_json('./data/{}_tweets.json'.format(filename), date_format='iso')
    tweets.to_csv('./data/{}_tweets.csv'.format(filename), sep='\t')
    
    users = pd.DataFrame(users)
    users.reset_index(inplace=True)
    users.to_json('./data/{}_users.json'.format(filename), date_format='iso')
    users.to_csv('./data/{}_users.csv'.format(filename), sep='\t')
    
    tweets_prepped = pd.DataFrame().assign(date=tweets['date'], id=str(tweets['id']), content=tweets['normalized'], followers=tweets['user_follower_count'], likeCount=tweets['likeCount'])
    
    #group_tweets_by_timewindow(tweets_prepped, filename)
    
scrap_tweets('loopring OR #lrc OR #loopring', '2021-10-30', '2021-10-31', 'test') 