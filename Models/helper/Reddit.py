
# importing the requests library
import requests
import json
import time
import math
import numpy as np
from datetime import timedelta
import praw
import pandas as pd
#from pmaw import PushshiftAPI
import datetime as dt
from psaw import PushshiftAPI
import os

config = {
    "username" : "bjorgbirb",
    "client_id" : "-QdhQYEx5aOdyl-p-UfoVQ",
    "client_secret" : "Zu1RswYSAKjFVABFb7AUTsKuAhg7IQ",
    "user_agent" : "script:https://github.com/bjorger/master-thesis:v1.0.0 (by u/bjorgbirb)"
}

reddit = praw.Reddit(client_id = config['client_id'], \
                     client_secret = config['client_secret'], \
                     user_agent = config['user_agent'])

api = PushshiftAPI()
start_epoch=int(dt.datetime(2021, 10, 30).timestamp())
end_epoch=int(dt.datetime(2022, 10, 31).timestamp())

subreddit="CryptoCurrency"
limit=1000000
filename='reddit_lrc_1year_unfiltered'
queries = ['loopring', 'LRC', '$LRC', 'Loopring']
submissions_stored = []

submissions = pd.DataFrame()
for query in queries:
    api_request_generator = api.search_submissions(q=query, limit=limit, after=start_epoch, before=end_epoch)
    # https://melaniewalsh.github.io/Intro-Cultural-Analytics/04-Data-Collection/14-Reddit-Data.html
    subs = []
    for submission in api_request_generator:
        submission = submission.d_
        
        if submission['id'] not in submissions_stored:
            submissions_stored.append(submission['id'])
            subs.append(submission)
        
        
    subs = pd.DataFrame(subs)
    submissions = submissions.append(pd.DataFrame(subs))

           
if not os.path.isfile('./data/{}.csv'.format(filename)):
    submissions.to_csv('./data/{}.csv'.format(filename), sep='\t')
    submissions.to_json('./data/{}.json'.format(filename), orient='records')
else: # else it exists so append without writing the header
    submissions.to_csv('./data/{}.csv'.format(filename), mode='a', header=False)
    submissions.to_json('./data/{}.json'.format(filename), orient='records')
