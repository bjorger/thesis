import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bi = pd.read_csv('./results/LSTM/best_sentiment_results_bidirectional.csv', sep=',')
stacked = pd.read_csv('./results/LSTM/best_sentiment_results_stacked.csv', sep=',')
single = pd.read_csv('./results/LSTM/best_sentiment_results_single.csv', sep=',')


"""
def get_mean_from_column(column_name: str, column_value):
    res = result[result[column_name] == column_value]
    print(res['rmse'].mean())

dropout_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for i in range(0, len(dropout_rate)):
    get_mean_from_column('dropout_rate', dropout_rate[i])
"""
"""
results_bi_reddit = bi[bi['name'].str.contains("Reddit")]
results_stacked_reddit = stacked[stacked['name'].str.contains("Reddit")]
results_single_reddit = single[single['name'].str.contains("Reddit")]
results_bi_twitter = bi[bi['name'].str.contains("Twitter")]
results_stacked_twitter = stacked[stacked['name'].str.contains("Twitter")]
results_single_twitter = single[single['name'].str.contains("Twitter")]
"""
def print_plots(results: pd.DataFrame, axis: plt.Axes, title: str):
    x_ticks = []
    axis.tick_params(axis='both', labelsize=8)
    for _, result in results.iterrows():
        name = result['name'].replace("Twitter", '').replace("_", ' ').replace("Price", 'Sentiment')
        rmse = result['rmse']
        x_ticks.append(name)
        axis.set_title('RMSE for {}'.format(title), rotation=0)
        axis.set_ylim(0, 0.01)
        axis.bar(name, rmse)
    axis.set_xticklabels(x_ticks, rotation=0)

"""
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))

  
print_plots(results_stacked_reddit, ax3, 'Stacked LSTM')
print_plots(results_bi_reddit, ax1, 'Bidirectional LSTM')
print_plots(results_single_reddit, ax2, 'Single Layer LSTM')
plt.savefig('Sentiment_Reddit_Variations.png')

print_plots(results_stacked_twitter, ax3, 'Stacked LSTM')
print_plots(results_bi_twitter, ax1, 'Bidirectional LSTM')
print_plots(results_single_twitter, ax2, 'Single Layer LSTM')
plt.savefig('Sentiment_Twitter_Variations.png')
"""


"""
reddit_sentiment = pd.read_csv('./test_data/lrc_1year_sentiment_reddit.csv', sep=',')
twitter_sentiment = pd.read_csv('./test_data/lrc_1year_sentiment_twitter.csv', sep=',')

reddit_mean = reddit_sentiment['Sentiment'].mean()
reddit_likes_mean = reddit_sentiment['Sentiment_Likes'].mean()
reddit_comments_mean = reddit_sentiment['Sentiment_amount_comments'].mean()
#twitter_mean = twitter_sentiment['Sentiment'].mean()

fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(15,15))

def process_sentiment_for_plot(sentiment_yearly: pd.Series):
    sentiment_daily = []
    sentiments = []
    day = 0
    for _, sentiment in sentiment_yearly.items():
        if day == 48:
            day = 0
            sentiments.append(np.array(sentiment_daily).mean())
            sentiment_daily = []
        sentiment_daily.append(sentiment)
        day += 1
        
    return sentiments
        
sentiments = process_sentiment_for_plot(reddit_sentiment['Sentiment'])
sentiments_likes = process_sentiment_for_plot(reddit_sentiment['Sentiment_Likes'])
sentiments_comments = process_sentiment_for_plot(reddit_sentiment['Sentiment_amount_comments'])

reddit_sentiments_processed = [
    {
        'sentiment': sentiments,
        'label': 'Sentiment',
    }, 
    {
        'sentiment': sentiments_likes, 
        'label': 'Sentiment Likes'}, 
    {
        'sentiment': sentiments_comments,
        'label': 'Sentiment Comments'
    }
]

reddit_sentiment_mean = [
    {
        'sentiment': reddit_mean,
        'label': 'Sentiment',
    }, 
    {
        'sentiment': reddit_likes_mean, 
        'label': 'Sentiment Likes'
    }, 
    {
        'sentiment': reddit_comments_mean,
        'label': 'Sentiment Comments'
    }
]

ax1.set_title('Sentiment Reddit (Bar Plot)', rotation=0)
ax1.tick_params(axis='both', labelsize=8)
ax1.set_ylabel("Sentiment")
for sentiment in reddit_sentiment_mean:
    ax1.bar(sentiment['label'], sentiment['sentiment'])

ax2.set_ylabel("Sentiment")
ax2.set_xlabel("Days of the year (31st October 21 - 31st October 22")
for sentiments in reddit_sentiments_processed:
    ax2.plot(sentiments['sentiment'], label=sentiments['label'])

plt.legend(loc="upper left")
plt.savefig('reddit_bias.png')


twitter_sentiment = pd.read_csv('./test_data/lrc_1year_sentiment_twitter.csv', sep=',')

twitter_mean = twitter_sentiment['Sentiment'].mean()
twitter_mean_followers = twitter_sentiment['Sentiment_Followers'].mean()
twitter_mean_likes = twitter_sentiment['Sentiment_Likes'].mean()

fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(15,15))

def process_sentiment_for_plot(sentiment_yearly: pd.Series):
    sentiment_daily = []
    sentiments = []
    day = 0
    for _, sentiment in sentiment_yearly.items():
        if day == 48:
            day = 0
            sentiments.append(np.array(sentiment_daily).mean())
            sentiment_daily = []
        sentiment_daily.append(sentiment)
        day += 1
        
    return sentiments
        
sentiments = process_sentiment_for_plot(twitter_sentiment['Sentiment'])
sentiments_followers = process_sentiment_for_plot(twitter_sentiment['Sentiment_Followers'])
sentiments_likes = process_sentiment_for_plot(twitter_sentiment['Sentiment_Likes'])

reddit_sentiments_processed = [
    {
        'sentiment': sentiments,
        'label': 'Sentiment',
    }, 
    {
        'sentiment': sentiments_followers, 
        'label': 'Sentiment Followers'
    }, 
    {
        'sentiment': sentiments_likes,
        'label': 'Sentiment Likes'
    }
]

reddit_sentiment_mean = [
    {
        'sentiment': twitter_mean,
        'label': 'Sentiment',
    }, 
    {
        'sentiment': twitter_mean_followers, 
        'label': 'Sentiment Followers'
    }, 
    {
        'sentiment': twitter_mean_likes,
        'label': 'Sentiment Likes'
    }
]

ax1.set_title('Sentiment Twitter (Bar Plot)', rotation=0)
ax1.set_ylabel("Sentiment")
ax1.tick_params(axis='both', labelsize=8)

for sentiment in reddit_sentiment_mean:
    ax1.bar(sentiment['label'], sentiment['sentiment'])

ax2.set_ylabel("Sentiment")
ax2.set_xlabel("Days of the year (31st October 21 - 31st October 22")
for sentiments in reddit_sentiments_processed:
    ax2.plot(sentiments['sentiment'], label=sentiments['label'])

plt.legend(loc="upper left")
plt.savefig('twitter_bias.png')
"""

results_single = pd.read_csv('./results/LSTM/results_single_used.csv', sep=',')
results_bi = pd.read_csv('./results/LSTM/results_bidirectional.csv', sep=',')
results_stacked = pd.read_csv('./results/LSTM/results_stacked.csv', sep=',')

"""
def get_grouped_results(df: pd.DataFrame, goal: str, eliminate: str):
    results_single = df[df['name'].str.contains(goal)]
    results_single = results_single[~results_single['name'].str.contains(eliminate)]
    return results_single.groupby(['name'], as_index=False).mean()

reddit_single_res = get_grouped_results(results_single, 'Reddit', 'Twitter')
reddit_bi_res = get_grouped_results(results_bi, 'Reddit', 'Twitter')
reddit_stacked_res = get_grouped_results(results_stacked, 'Reddit', 'Twitter')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,15))
print_plots(reddit_stacked_res, ax3, 'Stacked LSTM')
print_plots(reddit_bi_res, ax1, 'Bidirectional LSTM')
print_plots(reddit_single_res, ax2, 'Single Layer LSTM')
plt.subplots_adjust(hspace=0.3)
plt.show()
plt.savefig("reddit_architecture_comparison.png")
"""
"""
def get_grouped_results(df: pd.DataFrame, goal: str, eliminate: str):
    results_single = df[df['name'].str.contains(goal)]
    results_single = results_single[~results_single['name'].str.contains(eliminate)]
    results_single = results_single.append(df[df['name'].str.contains('Price_Twitter_Reddit_FNG_BTC')])
    return results_single.groupby(['name'], as_index=False).mean()

reddit_single_res = get_grouped_results(results_single, 'Twitter', 'Reddit')
reddit_bi_res = get_grouped_results(results_bi, 'Twitter', 'Reddit')
reddit_stacked_res = get_grouped_results(results_stacked, 'Twitter', 'Reddit')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15,15))
fig.tight_layout()
print_plots(reddit_stacked_res, ax3, 'Stacked LSTM')
print_plots(reddit_bi_res, ax1, 'Bidirectional LSTM')
print_plots(reddit_single_res, ax2, 'Single Layer LSTM')
plt.subplots_adjust(hspace=0.1)
plt.savefig("twitter_architecture_comparison.png")
"""

"""

def print_plots_neurons(results: pd.DataFrame, axis: plt.Axes, title: str):
    axis.tick_params(axis='both', labelsize=8)
    for _, result in results.iterrows():
        name = 'Neurons: {}'.format((''.join(map(str, result['neurons']))))
        rmse = result['rmse']
        axis.set_title('RMSE for {}'.format(title), rotation=0)
        axis.set_ylim(0, 0.015)
        axis.bar(name, rmse)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))

results_stacked = results_stacked.groupby(['neurons'], as_index=False).mean().sort_values(['rmse'], ascending=False)
results_bi = results_bi.groupby(['neurons'], as_index=False).mean().sort_values(['rmse'], ascending=False)

print(results_bi.head())
print(results_stacked.head())

print_plots_neurons(results_stacked, ax1, 'Stacked LSTM (Different Neurons)')
print_plots_neurons(results_bi, ax2, 'Bidirectional LSTM (Different Neurons)')
#plt.savefig('neurons_layered_architecture.png')
#plt.show()
"""


"""
fng = pd.read_csv('./test_data/fng_prepped.csv', sep=',')

def process_fng_for_plot(fng_yearly: pd.Series):
    fng_daily = []
    fngs = []
    day = 0
    for _, fng in fng_yearly.items():
        if day == 48:
            day = 0
            fngs.append(np.array(fng_daily).mean())
            fng_daily = []
        fng_daily.append(fng)
        day += 1
        
    return fngs

fng = process_fng_for_plot(fng['Value'])

plt.plot(fng)
plt.xlabel('Days', fontsize=8)
plt.ylabel('Fear and Greed Index', fontsize=8)
plt.title('Fear and Greed Index (31st October 2021 - 31st October 2022', fontsize=10)
plt.savefig('fear_and_greed.png')
"""
def print_horizontal_bars(results: pd.DataFrame, axis: plt.Axes, title: str):
    axis.tick_params(axis='both', labelsize=8)
    for _, result in results.iterrows():
        name = result['name'].replace('_', ' ')
        rmse = result['rmse']
        axis.set_title('RMSE for {}'.format(title), rotation=0)
        axis.set_xlim(0, 0.012)
        axis.barh(name, rmse)


results_single = pd.read_csv('./results/LSTM/results_single_used.csv', sep=',')
results_bi = pd.read_csv('./results/LSTM/results_bidirectional.csv', sep=',')
results_stacked = pd.read_csv('./results/LSTM/results_stacked.csv', sep=',')

results_bi = results_bi[results_bi['neurons'] == "128, 64"]
results_bi = results_bi.groupby(['name'], as_index=False).mean().sort_values(['rmse'], ascending=False)
results_stacked = results_stacked[results_stacked['neurons'] == "128, 64"]
results_stacked = results_stacked.groupby(['name'], as_index=False).mean().sort_values(['rmse'], ascending=False)
results_single = results_single.groupby(['name'], as_index=False).mean().sort_values(['rmse'], ascending=False)
fig, (ax1) = plt.subplots(1, 1, figsize=(15,15))
print_horizontal_bars(results_bi, ax1, 'Bidirectional Layer LSTM')
#plt.show()

print(results_bi.sort_values(['rmse']).iloc[0]['rmse'])
print(results_stacked.sort_values(['rmse']).iloc[0]['rmse'])
print(results_single.sort_values(['rmse']).iloc[0]['rmse'])
#plt.show()
plt.savefig('results_bidirectional_layer.png')
"""
ax1.set_title('RMSE Comparison for all architectures', rotation=0)
ax1.set_xlim(0, 0.006)
ax1.barh(['Stacked Layer', 'Bidirectional Layer', 'Single Layer'], [results_stacked.sort_values(['rmse']).iloc[0]['rmse'] ,results_bi.sort_values(['rmse']).iloc[0]['rmse'], results_single.sort_values(['rmse']).iloc[0]['rmse']])

plt.savefig('model_comparison.png')
#plt.show()
"""