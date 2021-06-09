import timeseries_to_gaf as ttg
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from multiprocessing import Pool
import pandas as pd
import os
import datetime as dt
from typing import *


PATH = os.path.dirname(__file__)
IMAGES_PATH = os.path.join(PATH, 'GramianAngularFields/TRAIN')
TEST_PATH = os.path.join(PATH, 'GramianAngularFields/TEST')
DATA_PATH = os.path.join(PATH, 'TimeSeries')


def data_to_image_preprocess() -> None:
    """
    :return: None
    """
    print('PROCESSING DATA')
    ive_data = 'IBM_adjusted.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(os.path.join(DATA_PATH, ive_data), names=col_name, header=None)
    # Drop unnecessary data_slice
    df = df.drop(['High', 'Low', 'Volume', 'Open'], axis=1)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    df = df.groupby(pd.Grouper(key='DateTime', freq='1h')).mean().reset_index()     # '1min'
    df['Close'] = df['Close'].replace(to_replace=0, method='ffill')
    # Remove non trading days and times
    clean_df = clean_non_trading_times(df)
    # Send to slicing
    set_gaf_data(clean_df)


def clean_non_trading_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Data with weekends and holidays
    :return trading_data:
    """
    # Weekends go out
    df = df[df['DateTime'].dt.weekday < 5].reset_index(drop=True)
    df = df.set_index('DateTime')
    # Remove non trading hours
    df = df.between_time('9:00', '16:00')
    df.reset_index(inplace=True)
    # Holiday days we want to delete from data_slice
    holidays = Calendar().holidays(start='2000-01-01', end='2020-12-31')
    m = df['DateTime'].isin(holidays)
    clean_df = df[~m].copy()
    trading_data = clean_df.fillna(method='ffill')
    return trading_data


def set_gaf_data(df: pd.DataFrame) -> None:
    """
    :param df: DataFrame data_slice
    :return: None
    """
    dates = df['DateTime'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = 20
    # Container to store data_slice for the creation of GAF
    decision_map = {key: [] for key in ['LONG', 'SHORT']}
    while True:
        if index >= len(list_dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['DateTime'] > list_dates[index - 20]) & (df['DateTime'] < list_dates[index])]
        gafs = []
        # Group data_slice by time frequency
        for freq in ['1h', '2h', '4h', '1d']:
            group_dt = data_slice.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['Close'].tail(20))
        # Decide what trading position we should take on that day
        future_value = df[df['DateTime'].dt.date.astype(str) == list_dates[index]]['Close'].iloc[-1]
        current_value = data_slice['Close'].iloc[-1]
        decision = trading_action(future_close=future_value, current_close=current_value)
        decision_map[decision].append([list_dates[index - 1], gafs])
        index += 1
    print('GENERATING IMAGES')
    # Generate the images from processed data_slice
    generate_gaf(decision_map)
    # Log stuff
    dt_points = dates.shape[0]
    total_short = len(decision_map['SHORT'])
    total_long = len(decision_map['LONG'])
    images_created = total_short + total_long
    print("========PREPROCESS REPORT========:\nTotal Data Points: {0}\nTotal Images Created: {1}"
          "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(dt_points,
                                                                           images_created,
                                                                           total_short,
                                                                           total_long))


def trading_action(future_close: int, current_close: int) -> str:
    """
    :param future_close: Integer
    :param current_close: Integer
    :return: Folder destination as String
    """
    current_close = current_close
    future_close = future_close
    if current_close < future_close:
        decision = 'LONG'
    else:
        decision = 'SHORT'
    return decision


def generate_gaf(images_data: Dict[str, pd.DataFrame]) -> None:
    """
    :param images_data:
    :return:
    """
    for decision, data in images_data.items():
        for image_data in data:
            to_plot = [ttg.create_gaf(x)['gadf'] for x in image_data[1]]
            ttg.create_images(X_plots=to_plot,
                              image_name='{0}'.format(image_data[0].replace('-', '_')),
                              destination=decision)


if __name__ == "__main__":
    pool = Pool(os.cpu_count())
    print(dt.datetime.now())
    print('CONVERTING TIME-SERIES TO IMAGES')
    pool.apply(data_to_image_preprocess)
    print('DONE!')
    print(dt.datetime.now())
