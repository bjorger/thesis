import pandas as pd
from datetime import timedelta, datetime

posts = pd.read_json('data/test_data/fear_and_greed.json', orient="records")

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
    analyzed_posts = pd.DataFrame(columns=['Timewindow', 'Value'])

    for year in range(2021, datetime.now().year + 1):
        posts_year = posts[posts['timestamp'].dt.year == year]
        for month in months:
            posts_month = posts_year[pd.to_datetime(posts_year['timestamp']).dt.strftime('%B') == month]
            for day in range(1, months[month] + 1):
                posts_day = posts_month.loc[(posts_month['timestamp'].dt.day == day)]
                val = posts_day['value']

                if len(val) == 0:
                    continue
                
                for i in range(1, 49):
                    start = str(timedelta(minutes=(i-1)*30))
                    end = str(timedelta(minutes=(i*30)-1, seconds=59))
                    timewindow = '{}-{}-{} {}-{}'.format(day, month, year, start, end)
                    
                    entry = {
                        'Timewindow': timewindow,
                        'Value': val.values[0]
                    }
                    analyzed_posts = analyzed_posts.append(entry, ignore_index=True)

    analyzed_posts.to_csv('data/test_data/{}.csv'.format(filename), sep=',')
    analyzed_posts.to_json('data/test_data/{}.json'.format(filename), orient="records")

group_by_timewindow(posts, "fng_prepped")