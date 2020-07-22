import pandas as pd
import dask.dataframe as dd

PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'

def frequency_format(df, frequency):
    group_dt = df.groupby(pd.Grouper(key='DateTime', freq=frequency)).sum().reset_index()
    group_dt['Open'] = group_dt['Open'].replace(to_replace=0, method='ffill')
    return group_dt


if __name__ == "__main__":
    ive_data = 'IVE_tickbidask.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Volume']
    df = pd.read_csv(PATH + ive_data, names=col_name, header=None)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    time_series =  {}
    for tg in ['1h', '8h']: #'2h', '4h', '1d']:
        frequency_format(df, tg).to_csv(PATH + 'ts_{0}_{1}.csv'.format('ive', tg), index=False)

