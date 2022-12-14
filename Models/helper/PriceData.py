from MongoDB import MongoDB
import pandas as pd
import requests
from datetime import date, timedelta
import pandas as pd
import os

#API_KEY_1 = "E57C556D-3D28-4385-91DC-83E8417B0A26"
"""
url = 'https://rest.coinapi.io/v1/exchangerate/LRC/USD/history?period_id=30MIN&time_start=2017-10-31T00:00:00&time_end=2022-10-31T00:00:00'
headers = {'X-CoinAPI-Key' : API_KEY}
response = requests.get(url, headers=headers)

res_df = pd.DataFrame(response.json())

res_df.to_csv('./data/api_test.csv', sep=',')
"""

#API_KEY_2 = "2ED9FA19-9282-48FA-BE26-16BC15668597"
"""
filename = './data/lrc_coin_api.csv'
start_date = date(2019, 12, 25)
end_date = date(2022, 10, 31)
delta = end_date - start_date
print(delta.days)
for i in range(0, delta.days, 2):
    start = start_date + timedelta(minutes=i * 48 * 30)
    end_date = start_date + timedelta(minutes=(i + 2) * 48 * 30 if i > 0 else 48 * 30)
    start = start.strftime("%Y-%m-%dT%H:%M:%S")
    end = end_date.strftime("%Y-%m-%dT%H:%M:%S")
    
    url = 'https://rest.coinapi.io/v1/exchangerate/LRC/USD/history?period_id=30MIN&time_start={}&time_end={}'.format(start, end)
    headers = {'X-CoinAPI-Key' : API_KEY_3}
    response = requests.get(url, headers=headers)
    print(response.status_code)
    res_df = pd.DataFrame(response.json())
    
    if not os.path.isfile(filename):
        res_df.to_csv(filename, sep=',')
    else: # else it exists so append without writing the header
        res_df.to_csv(filename, mode='a', header=False)
""" 
"""
url = 'https://rest.coinapi.io/v1/exchangerate/LRC/USD/history?period_id=30MIN&time_start=2017-11-04T03:30:00&time_end=2022-10-31T00:00:00'
headers = {'X-CoinAPI-Key' : API_KEY}
response = requests.get(url, headers=headers)

res_df = pd.DataFrame(response.json())

res_df.to_csv('./data/api_test.csv', sep=',')
"""
"""
mong = MongoDB('lrc_price_snapshots')
df = pd.DataFrame(list(mong.collection.find()))

df.to_csv('./data/lrc_snapshots.csv')
"""
#API_KEY="2ED9FA19-9282-48FA-BE26-16BC15668597"
API_KEY= "E57C556D-3D28-4385-91DC-83E8417B0A26"

#API_KEY = "C74EBB92-3668-43C2-83D5-7FE045DCB7C1"

filename = './sentiment/data/btc_1year_coin_api'
start_date = date(2021, 10, 30)
end_date = date(2022, 10, 31)

url = 'https://rest.coinapi.io/v1/exchangerate/BTC/USD/history?period_id=30MIN&time_start={}&time_end={}&limit=100000'.format(start_date.strftime("%Y-%m-%dT%H:%M:%S"), end_date.strftime("%Y-%m-%dT%H:%M:%S"))
headers = {'X-CoinAPI-Key' : API_KEY}
response = requests.get(url, headers=headers)
print(response.status_code)
res_df = pd.DataFrame(response.json())
res_df.to_csv('{}.csv'.format(filename), sep=',')
res_df.to_json('{}.json'.format(filename), orient='records')


"""
start_date = date(2021, 10, 30)
end_date = date(2022, 10, 30)

print(start_date.strftime("%Y-%m-%dT%H:%M:%S"))
print(end_date.strftime("%Y-%m-%dT%H:%M:%S"))

print(end_date - start_date)
"""