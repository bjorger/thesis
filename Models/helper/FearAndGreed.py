import requests
import pandas as pd
from pandas import json_normalize

fng = "https://api.alternative.me/fng/?limit=0"
response = requests.get(fng)
res_dict = response.json()
res_dict = json_normalize(res_dict['data'])
res_df = pd.DataFrame(res_dict)

res_df['timestamp'] = pd.to_datetime(res_df['timestamp'], unit="s")
res_df.set_index(res_df['timestamp'], inplace=True)
res_df.index = res_df['timestamp']
startDate = pd.to_datetime('2021-10-31')
endDate = pd.to_datetime('2022-10-31')

res_df = res_df[res_df.index >= startDate]
res_df = res_df[res_df.index <= endDate]
res_df = res_df.iloc[::-1]
print(res_df['timestamp'])
res_df.to_csv('sentiment/data/test_data/fear_and_greed.csv', sep=',')
res_df.to_json('sentiment/data/test_data/fear_and_greed.json', orient='records')