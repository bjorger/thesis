
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
import re
from TweetNormalizer import normalizeTweet

columns_to_delete = [
    "all_awardings", 
    "allow_live_comments", 
    "author_flair_css_class", 
    "author_flair_richtext", 
    "author_flair_text", 
    "author_flair_type", 
    "author_fullname", 
    "author_is_blocked", 
    "author_patreon_flair", 
    "author_premium", 
    "awarders", 
    "can_mod_post", 
    "contest_mode", 
    "domain", 
    "full_link", 
    "gildings", 
    "is_created_from_ads_ui", 
    "is_crosspostable", 
    "is_meta", 
    "is_original_content", 
    "is_reddit_media_domain", 
    "is_robot_indexable", 
    "is_self", 
    "is_video", 
    "link_flair_background_color", 
    "link_flair_richtext", 
    "link_flair_text_color", 
    "link_flair_type", 
    "locked", 
    "media_only", 
    "no_follow", 
    "num_crossposts", 
    "over_18", 
    "permalink", 
    "pinned", 
    "post_hint", 
    "preview", 
    "retrieved_on", 
    "send_replies", 
    "spoiler", 
    "stickied", 
    "subreddit", 
    "subreddit_id", 
    "thumbnail", 
    "total_awards_received", 
    "treatment_tags", 
    "url", 
    "parent_whitelist_status", 
    "pwls", 
    "removed_by_category", 
    "url_overridden_by_dest", 
    "whitelist_status", 
    "wls", 
    "link_flair_template_id", 
    "link_flair_text", 
    "thumbnail_height", 
    "thumbnail_width", 
    "author_flair_template_id", 
    "author_flair_text_color", 
    "suggested_sort", 
    "author_flair_background_color", 
    "distinguished", 
    "link_flair_css_class", 
    "media", 
    "media_embed", 
    "secure_media", 
    "secure_media_embed", 
    "gallery_data", 
    "is_gallery", 
    "media_metadata", 
    "crosspost_parent", 
    "crosspost_parent_list", 
    "poll_data", 
    "author_cakeday", 
    "banned_by", 
    "discussion_type", 
    "edited", 
    "call_to_action", 
    "domain_override", 
    "events", 
    "eventsOnRender", 
    "href_url", 
    "is_blank", 
    "mobile_ad_url", 
    "outbound_link", 
    "promoted", 
    "show_media", 
    "third_party_trackers", 
    "gilded", 
    "collections", 
    "content_categories", 
    "event_end", 
    "event_is_live", 
    "event_start"
]
"""
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
start_epoch=int(dt.datetime(2022, 10, 30).timestamp())
end_epoch=int(dt.datetime(2022, 10, 31).timestamp())

subreddit="CryptoCurrency"
filename='reddit_btc_1year_unfiltered'
queries = ['bitcoin', 'BTC', '$BTC', 'Bitcoin']
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
            for key in columns_to_delete:
                del submission[key]
            subs.append(submission)
        
    
    subs = pd.DataFrame(subs)
    submissions = submissions.append(pd.DataFrame(subs))

submissions.to_csv('./data/{}.csv'.format(filename), sep='\t')
submissions.to_json('./data/{}.json'.format(filename), orient='records')
"""
"""
reddit = pd.read_json('sentiment/data/final/reddit_lrc_1year.json', orient="records")
"""

"""
for key in columns_to_delete:
    del reddit[key]

for _, post in reddit.iterrows():
    if post['selftext']:
        post['selftext'] = post['selftext'].replace('\n', '')
        post['selftext'] = re.sub("@[A-Za-z0-9_]+","", post['selftext'])
        post['selftext'] = re.sub("#[A-Za-z0-9_]+","", post['selftext'])
        post['selftext'] = normalizeTweet(post['selftext'])

reddit.to_csv('sentiment/data/test_data/reddit.csv', sep=',')
reddit.to_json('sentiment/data/test_data/reddit.json', orient='records')
"""

reddit = pd.read_json('sentiment/data/test_data/reddit.json', orient="records")
"""
reddit_cleaned = []
for _, post in reddit.iterrows():
    if post['selftext']:
        post['selftext'] = post['selftext'].replace('\n', '')
        post['selftext'] = re.sub("@[A-Za-z0-9_]+","", post['selftext'])
        post['selftext'] = re.sub("#[A-Za-z0-9_]+","", post['selftext'])
        post['selftext'] = normalizeTweet(post['selftext'])
    reddit_cleaned.append(post)
"""
reddit_cleaned = []
for _, post in reddit.iterrows():
    post['date'] = pd.to_datetime(post['created_utc'], unit="s")
    reddit_cleaned.append(post)
reddit = pd.DataFrame(reddit_cleaned)
reddit = reddit.iloc[::-1]
reddit.to_csv('sentiment/data/test_data/reddit.csv', sep='\t')
reddit.to_json('sentiment/data/test_data/reddit.json', orient='records')
