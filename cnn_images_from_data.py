import timeseries_to_gaf as ttg
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
PATH = os.path.join(os.path.dirname(__file__), 'TimeSeries')
# PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'
rows_size = 40

def wrapper(args):
   return generate_time_series_image(*args)

def create_gaf(df):
    """
    :param df:
    :return:
    """
    return ttg.create_gaf(df)['gadf']



def split_dataframes_by_row(df, splits):
    """
    :param df: DataFrame
    :param splits: Number of splits i
    :return: List of DataFrames
    """
    list_df = np.split(df, range(splits, len(df), splits))
    last_df_size = list_df[-1].shape[0]
    if last_df_size < rows_size:  #20
        list_df = list_df[:-1]
    return list_df


def buy_hold_sell(time_piece):
    """
    :param time_piece: DataFrame
    :return: DataFrame and all but last row of data
    """
    when_selling = time_piece[-1]
    when_buying = time_piece[0]
    if when_buying < when_selling:
        buy_sell_hold = 'SELL'
    else:
        buy_sell_hold = 'BUY'
    return buy_sell_hold, time_piece[:-1]


def generate_time_series_image(intervals, inter_raw):
    """
    :param intervals: Time interval
    :param inter_raw: Main time table
    """
    interval_raw = inter_raw
    time_intervals = intervals
    for df in time_intervals:
        min_date = df.index.min()
        max_date = df.index.max()
        full_time_series_gaf = {}
        invest_decision, when_sell = buy_hold_sell(time_piece=df)
        full_time_series_gaf[min_date + ' ' + max_date] = [[create_gaf(when_sell)]]
        for data in interval_raw:
            data_slice = data.loc[(data.index > min_date) & (data.index < max_date)]
            full_time_series_gaf[min_date + ' ' + max_date].append(
                [create_gaf(d) for d in split_dataframes_by_row(data_slice, rows_size)])
        full_time_series_gaf[min_date + ' ' + max_date].sort(key=len)
        num = 0
        for i in full_time_series_gaf[min_date + ' ' + max_date][0]:
            for r in full_time_series_gaf[min_date + ' ' + max_date][1]:
                for g in full_time_series_gaf[min_date + ' ' + max_date][2]:
                    for q in full_time_series_gaf[min_date + ' ' + max_date][3]:
                        ttg.create_images(X_plots=[i, r, g, q],
                                          image_name='{0}_{1}_{2}'.format(min_date, max_date, num),
                                          destination=invest_decision)
                        # print('GENERATING GAF FOR {}'.format('{0}_{1}_{2}'.format(min_date, max_date, num)))
                        num += 1
        # print('COMPLETED GENERATING GAF FOR {0}, TOTAL NUM: {1}'.format('{0}_{1}'.format(min_date, max_date), num))
        full_time_series_gaf[min_date + ' ' + max_date] = None


def split_time_series(init_df, raw_dfs):
    """
    :param init_df:
    :param raw_dfs:
    :return:
    """
    daily_df = pd.read_csv(os.path.join(PATH, init_df), index_col='DateTime',
                             dtype={'Volume': float, 'Open': float, 'High': float, 'Low': float})['Open']
    first_interval = split_dataframes_by_row(daily_df, rows_size)
    intervals = []
    for file_name in raw_dfs:
        intervals.append(pd.read_csv(os.path.join(PATH, file_name), index_col='DateTime',
                                        dtype={'Volume': float, 'Open': float, 'High': float, 'Low': float})['Open'])
    return first_interval, intervals



if __name__ == "__main__":
    first_file = 'ts_ive_1d.csv'
    files_relationship = ['ts_ive_8h.csv', 'ts_ive_4h.csv', 'ts_ive_2h.csv']
    first_interval_unedited, intervals_unedited = split_time_series(first_file, files_relationship)
    import datetime
    pool = Pool(4)
    print(datetime.datetime.now())
    pool.map(wrapper,[(first_interval_unedited, intervals_unedited)])
    print(datetime.datetime.now())
    # print(datetime.datetime.now())
    # generate_time_series_image(first_interval_unedited, intervals_unedited)
    # print(datetime.datetime.now())
