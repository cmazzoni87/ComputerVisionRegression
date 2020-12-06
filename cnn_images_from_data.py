import timeseries_to_gaf as ttg
import construct_time_series as cts
import pandas as pd
import matplotlib as mpl
import numpy as np
import dask.dataframe as dd


PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'


def create_gaf(df):
    """
    :param df:
    :return:
    """
    return ttg.create_gaf(df, rows_size)['gadf']


def split_dataframes(df, splits):
    """
    :param df:
    :param splits:
    :return:
    """
    list_df = np.split(df, range(splits, len(df), splits))
    last_df_size = list_df[-1].shape[0]
    if last_df_size < rows_size:  #20
        list_df = list_df[:-1]
    return list_df


# I SHOULD ADD COMPLEXITY HERE, FOR NOW NEED THE MACHINE TO LEARN PATTERS ON IT'S OWN REGARDLESS OF TECHNICAL ANALYSIS
def buy_hold_sell(time_piece, entire_df, windows=(10, 20)):
    """
    :param time_piece:
    :param entire_df:
    :param windows:
    :return:
    """
    when_selling = time_piece[-1]
    when_buying = time_piece[0]
    if when_buying < when_selling:
        buy_sell_hold = 'SELL'
    else:
        buy_sell_hold = 'BUY'
    # else:
    #     buy_sell_hold = 'HOLD'
    return buy_sell_hold


def generate_time_series_image(intervals, inter_raw, daily_df):
    """
    :param intervals:
    :param inter_raw:
    :param daily_df:
    :return:
    """
    interval_raw = inter_raw
    time_intervals = intervals
    daily = daily_df
    for df in time_intervals:
        min_date = df.index.min()
        max_date = df.index.max()
        full_timeseries_gaf = {}
        invest_decition = buy_hold_sell(time_piece=df, entire_df=daily)
        full_timeseries_gaf[min_date + ' ' + max_date] = [[create_gaf(df)]]
        for data in interval_raw:
            data_slice = data.loc[(data.index > min_date) & (data.index < max_date)]
            full_timeseries_gaf[min_date + ' ' + max_date].append(
                [create_gaf(d) for d in split_dataframes(data_slice, rows_size)])
        full_timeseries_gaf[min_date + ' ' + max_date].sort(key=len)
        num = 0
        for i in full_timeseries_gaf[min_date + ' ' + max_date][0]:
            for r in full_timeseries_gaf[min_date + ' ' + max_date][1]:
                for g in full_timeseries_gaf[min_date + ' ' + max_date][2]:
                    for q in full_timeseries_gaf[min_date + ' ' + max_date][3]:
                        ttg.create_images(X_plots=[i, r, g, q],
                                          image_name='{0}_{1}_{2}'.format(min_date, max_date, num),
                                          destination=invest_decition)
                        print('GENERATING GAF FOR {}'.format('{0}_{1}_{2}'.format(min_date, max_date, num)))
                        num += 1
        print('COMPLETED GENERATING GAF FOR {0}, TOTAL NUM: {1}'.format('{0}_{1}'.format(min_date, max_date), num))
        full_timeseries_gaf[min_date + ' ' + max_date] = None


def split_time_series(init_df, raw_dfs):
    """
    :param init_df:
    :param raw_dfs:
    :return:
    """
    daily_df = pd.read_csv(PATH + init_df, index_col='DateTime',
                             dtype={'Volume': float, 'Open': float, 'High': float, 'Low': float})['Open']
    # split_interval = first_file.split('_')[2].split('.')[0]
    first_interval = split_dataframes(daily_df, rows_size)
    intervals = []
    for file_name in raw_dfs:
        intervals.append(pd.read_csv(PATH + file_name, index_col='DateTime',
                                        dtype={'Volume': float, 'Open': float, 'High': float, 'Low': float})['Open'])
    return daily_df, first_interval, intervals


if __name__ == "__main__":
    rows_size = 5
    first_file = 'ts_ive_1d.csv'
    files_relationship = ['ts_ive_8h.csv', 'ts_ive_4h.csv', 'ts_ive_2h.csv']
    daily_data, first_interval_unedited, intervals_unedited = split_time_series(first_file, files_relationship)
    generate_time_series_image(first_interval_unedited, intervals_unedited, daily_data)
