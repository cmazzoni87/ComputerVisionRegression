import pandas as pd
import os
PATH = os.path.join(os.path.dirname(__file__), 'TimeSeries')

def frequency_format(df, frequency):
    """
    :param df: DataFrame
    :param frequency: String time frequency used in method
    :return: New grouped by time frequency DataFrame
    """
    group_dt = df.groupby(pd.Grouper(key='DateTime', freq=frequency)).mean().reset_index()
    group_dt['Open'] = group_dt['Open'].fillna(0)
    group_dt = group_dt[group_dt['DateTime'].dt.weekday < 5].reset_index(drop=True)
    return group_dt

if __name__ == "__main__":
    ive_data = 'IVE_tickbidask.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Volume']
    df = pd.read_csv(os.path.join(PATH, ive_data), names=col_name, header=None)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    time_series = {}
    for tg in ['1h', '8h','2h', '4h', '1d']:
        frequency_format(df, tg).to_csv(os.path.join(PATH, 'ts_{0}_{1}.csv'.format('ive', tg)), index=False)
