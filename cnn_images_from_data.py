import timeseries_to_gaf as ttg
import construct_time_series as cts
import pandas as pd
import matplotlib as mpl
import numpy as np
import dask.dataframe as dd

PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'

def create_gaf(df):
    return ttg.create_gaf(df)['gadf']

def split_dataframes(df, splits):
    list_df = np.split(df, range(splits, len(df), splits))
    last_df_size = list_df[-1].shape[0]
    if last_df_size < 20:
        list_df = list_df[:-2]
    return list_df

if __name__ == "__main__":
    rows_size = 20
    first_file = 'ts_ive_1d.csv'
    files_relationship = ['ts_ive_4h.csv', 'ts_ive_2h.csv', 'ts_ive_30min.csv']
    daily_data = pd.read_csv(PATH + first_file, index_col='DateTime',
                             dtype={'Volume': float, 'Open': float, 'High': float, 'Low':float})['Open']
    tag_date = first_file.split('_')[2].split('.')[0]
    time_intervals = split_dataframes(daily_data, rows_size)
    interval_raw = []
    for file_name in files_relationship:
        interval_raw.append(pd.read_csv(PATH + file_name, index_col='DateTime',
                                        dtype={'Volume': float, 'Open': float, 'High': float, 'Low':float})['Open'])
    for df in time_intervals:
        min_date = df.index.min()
        max_date = df.index.max()
        full_timeseries_gaf = {}
        full_timeseries_gaf[min_date + ' ' + max_date] = [[create_gaf(df)]]
        for data in interval_raw:
            data_slice = data.loc[(data.index > min_date) & (data.index < max_date)]
            full_timeseries_gaf[min_date + ' ' + max_date].append([create_gaf(d) for d in split_dataframes(data_slice, rows_size)])
        full_timeseries_gaf[min_date + ' ' + max_date].sort(key=len)
        num = 0
        for i in full_timeseries_gaf[min_date + ' ' + max_date][0]:
            for r in full_timeseries_gaf[min_date + ' ' + max_date][1]:
                for g in full_timeseries_gaf[min_date + ' ' + max_date][2]:
                    for q in full_timeseries_gaf[min_date + ' ' + max_date][3]:
                        ttg.create_images([i, r, g, q], '{0}_{1}_{2}'.format(min_date, max_date, num))
                        print('COMPLETED GENERATING GAF FOR {}'.format('{0}_{1}_{2}'.format(min_date, max_date, num)))
                        num += 1
        full_timeseries_gaf[min_date + ' ' + max_date] = None
