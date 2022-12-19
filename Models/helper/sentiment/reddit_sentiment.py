from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import timedelta, datetime
"""
This updated version of the script uses a more elaborate approach to adjusting the sentiment score based on the number of likes and amount_comments. In this approach, the sentiment score is adjusted based on the number of likes and amount_comments, but with a maximum adjustment of 10 (i.e. a post with 10,000 likes or 10,000 amount_comments will not receive any further adjustment to the sentiment score). This helps to prevent extremely large values of likes or amount_comments from having an overly large influence on the sentiment score.

Again, this is just an example and the specific approach to adjusting the sentiment score may need to be fine-tuned for your specific use case.
"""
def calculate_post_sentiment_with_likes_and_follower(post, likes, amount_comments):
    # Use VADER to calculate the sentiment of the post
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(post)

    # Calculate the sentiment score
    sentiment_score = sentiment['compound']

    # Adjust the sentiment score based on the number of likes and amount_comments
    weighted_sentiment = sentiment_score * (likes + 1) * (amount_comments + 1)
    return weighted_sentiment

def calculate_post_sentiment_with_likes(post, likes):
    # Use VADER to calculate the sentiment of the post
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(post)

    # Calculate the sentiment score
    sentiment_score = sentiment['compound']

    # Adjust the sentiment score based on the number of likes
    if likes > 0:
        sentiment_score *= (1 + min(likes, 10000) / 1000)

    # Keep the sentiment score in the range of -1 to 1
    #sentiment_score = min(1, max(-1, sentiment_score))

    return sentiment_score

def calculate_sentiment(post): 
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(post)

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

def group_by_timewindow(posts: pd.DataFrame, filename: str) -> pd.DataFrame:
    posts['date'] = pd.to_datetime(posts['date'])
    posts.set_index('date')
    posts.index = posts['date']
    startDate = pd.to_datetime('2021-10-31 00:00:00')

    posts = posts[posts.index >= startDate]
    
    analyzer = SentimentIntensityAnalyzer()

    analyzed_posts = pd.DataFrame(columns=['Timewindow', 'Sentiment', 'Volume'])

    likeWeight = 0.2
    followerWeight = 0.4

    for year in range(2021, datetime.now().year + 1):
        posts_year = posts[posts['date'].dt.year == year]
        for month in months:
            posts_month = posts_year[pd.to_datetime(posts_year['date']).dt.strftime('%B') == month]
            for day in range(1, months[month] + 1):
                posts_day = posts_month.loc[(posts_month['date'].dt.day == day)]

                if (posts_day.size == 0):
                    continue
                
                for i in range(1, 49):
                    start = str(timedelta(minutes=(i-1)*30))
                    end = str(timedelta(minutes=(i*30)-1, seconds=59))
                    current_posts = posts_day.between_time(start, end)
                    results_post = []
                    results_post_likes = []
                    results_post_amount_comments = []
                    posts_in_timeframe = []
                    
                    for _, post in current_posts.iterrows():
                        post_id = str(post['id'])
                        if (post_id in posts_in_timeframe):
                            continue
                        text = post['selftext']
                        if not post['selftext']:
                            text = post['title']
                            
                        sentiment_result_likes = calculate_post_sentiment_with_likes(text, int(post['score']))
                        sentiment_result = calculate_sentiment(text)
                        sentiment_result_amount_comments = calculate_post_sentiment_with_likes(text, post['num_comments'])
                             
                        results_post.append(sentiment_result)
                        results_post_likes.append(sentiment_result_likes)
                        results_post_amount_comments.append(sentiment_result_amount_comments)
                        posts_in_timeframe.append(post_id)
                    
                    sentiment = 0
                    sentiment_likes = 0
                    sentiment_amount_comments = 0
                    
                    timewindow = '{}-{}-{} {}-{}'.format(day, month, year, start, end)

                    if (len(results_post) > 0):
                        sentiment = sum(results_post) / len(results_post)
                        sentiment_likes = sum(results_post_likes) / len(results_post)
                        sentiment_amount_comments = sum(results_post_amount_comments) / len(results_post)

                    entry = {
                        'Timewindow': timewindow,
                        'Sentiment': sentiment,
                        'Sentiment_Likes': sentiment_likes,
                        'Sentiment_amount_comments': sentiment_amount_comments,
                        'Volume': len(current_posts),
                        'postIds': ','.join(posts_in_timeframe),
                    }
                    analyzed_posts = analyzed_posts.append(entry, ignore_index=True)
                    
    analyzed_posts.to_csv('data/test_data/{}.csv'.format(filename), sep=',')
    analyzed_posts.to_json('data/test_data/{}.json'.format(filename), orient="records")
        



"""
posts = pd.read_json('data/lrc_1year_unfiltered_posts.json', orient="records")
users = pd.read_json('data/lrc_1year_unfiltered_users.json', orient="records")

numComments = []
for _, post in posts.iterrows():
    user = users.loc[users['username'] == post['user']]
    numComments.append(user['amount_commentsCount'])
    
posts_prepped = pd.DataFrame().assign(date=posts['date'], id=str(posts['id']), selftext=posts['normalized'], numComments=numComments, likeCount=posts['likeCount'])

posts_prepped.to_csv('data/lrc_1year_posts_with_amount_comments.csv', sep='\t')
posts_prepped.to_json('data/lrc_1year_posts_with_amount_comments.json', orient="records")
"""

posts = pd.read_json('data/test_data/reddit.json', orient="records")

group_by_timewindow(posts, "lrc_1year_sentiment_reddit_likes_greater_one")