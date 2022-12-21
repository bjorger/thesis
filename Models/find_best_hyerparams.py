import pandas as pd

result = pd.read_csv('./results/LSTM/results_stacked.csv', sep=',')


def get_mean_from_column(column_name: str, column_value):
    res = result[result[column_name] == column_value]
    print(res['rmse'].mean())

dropout_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for i in range(0, len(dropout_rate)):
    get_mean_from_column('dropout_rate', dropout_rate[i])