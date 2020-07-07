import timeseries_to_gaf as ttg
import construct_time_series as cts
import pandas as pd
import matplotlib as mpl
import numpy as np

PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'

def split_dataframes(df, splits):
    return np.split(df, range(splits, len(df), splits))

def start_ts_freq_gen(ive_data, col_name, frequency, tag):
    df = pd.read_csv(PATH + ive_data, names=col_name, header=None)
    # May the gods of programming forgive this literal statement
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    ts_dict = {}
    for tg in frequency:
        freq_df = cts.frequency_format(df, tg)
        freq_df[['DateTime', 'Open', 'Volume']].to_csv(PATH + 'ts_{0}_{1}.csv'.format(tag, frequency), index=False)
        ts_dict[tg] = split_dataframes(freq_df[['DateTime', 'Open', 'Volume']], 20)
    return ts_dict


if __name__ == "__main__":
    TAG = 'ive'
    IVE_DATA = 'IVE_tickbidask.txt'
    COL_NAME = ['Date', 'Time', 'Open', 'High', 'Low', 'Volume']
    FREQ = ['8h', '30min', '1h', '4h', '1d']
    ts_df = start_ts_freq_gen(ive_data=IVE_DATA, col_name=COL_NAME, frequency=FREQ, tag=TAG)