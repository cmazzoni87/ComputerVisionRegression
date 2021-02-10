import timeseries_to_gaf as ttg
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from multiprocessing import Pool
import pandas as pd
import os
import datetime as dt
import shutil
import glob

PATH = os.path.dirname(__file__)
IMAGES_PATH = os.path.join(PATH , 'GramianAnagularFields/TRAIN')
TEST_PATH = os.path.join(PATH , 'GramianAnagularFields/TEST')
DATA_PATH = os.path.join(PATH, 'TimeSeries')

def data_to_image_preprocess():
    """
    :return: None
    """
    print('PROCESSING DATA')
    ive_data = 'IVE_tickbidask.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Volume']
    df = pd.read_csv(os.path.join(DATA_PATH, ive_data), names=col_name, header=None)
    # Drop unnecessary data
    df = df.drop(['High', 'Low', 'Volume'], axis=1)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    df = df.groupby(pd.Grouper(key='DateTime', freq='1h')).mean().reset_index()    #'1min'
    df['Open'] = df['Open'].replace(to_replace=0, method='ffill')
    # Remove non trading days and times
    clean_df = clean_non_trading_times(df)
    # Send to slicing
    set_gaf_data(clean_df)

def clean_non_trading_times(df):
    """
    :param df: Data with weekends and holidays
    :return trading_data:
    """
    # Weekends go out
    df = df[df['DateTime'].dt.weekday < 5].reset_index(drop=True)
    df = df.set_index('DateTime')
    # Remove non trading hours
    df = df.between_time('9:00','16:00')
    df.reset_index(inplace=True)
    # Holiday days we want to delete from data
    holidays = calendar().holidays(start='2000-01-01', end='2020-12-31')
    m = df['DateTime'].isin(holidays)
    clean_df = df[~m].copy()
    trading_data = clean_df.fillna(method='ffill')
    return trading_data

def set_gaf_data(df):
    """
    :param df: DataFrame data
    :return: None
    """
    dates = df['DateTime'].dt.date
    dates = dates.drop_duplicates()
    list_dates = dates.apply(str).tolist()
    index = 20
    # Container to store data for the creation of GAF
    decision_map = {key: [] for key in ['LONG', 'SHORT']}
    while True:
        if index >= len(list_dates) - 1:
            break
        # Select appropriate timeframe
        data_slice = df.loc[(df['DateTime'] > list_dates[index - 20]) & (df['DateTime'] < list_dates[index])]
        gafs = []
        # Group data by time frequency
        for freq in ['1h', '2h', '4h', '1d']:
            group_dt = data_slice.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index()
            group_dt = group_dt.dropna()
            gafs.append(group_dt['Open'].tail(20))
        # Decide what trading position we should take on that day
        decision = trading_action(data=df, index=list_dates[index])
        decision_map[decision].append(gafs)
        index += 1
    print('GEMERATING IMAGES')
    # Generate the images from processed data
    generate_gaf(decision_map)
    # Log stuff
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    dt_points = dates.shape[0]
    total_short = len(decision_map['SHORT'])
    total_long = len(decision_map['LONG'])
    images_created = total_short + total_long
    f = open(os.path.join(PATH, 'preprocessing_summary_{}.txt'.format(timestamp)), 'w')
    f.write("========PREROCESS REPORT========:\nTotal Datapoints: {0}\nTotal Images Created: {1}"
            "\nTotal LONG positions: {2}\nTotal SHORT positions: {3}".format(dt_points,
                                                                             images_created,
                                                                             total_short,
                                                                             total_long))
    f.close()
    cup_of_test_data()

def cup_of_test_data():
    """
    :return: None
    """
    long_path = os.path.join(IMAGES_PATH, 'LONG')
    short_path = os.path.join(IMAGES_PATH, 'SHORT')
    short = glob.glob(short_path + '/*', recursive=False)
    long = glob.glob(long_path + '/*', recursive=False)
    # TAKE LAST 20 ROWS OF EACH FOLDER FOR TESTING
    test_files = long[-21:-1] + short[-21:-1]
    for files in test_files:
        source_tag = files.split('\\')[-2]
        file_name = files.split('\\')[-1]
        file_new_name = source_tag + '\\' + file_name
        shutil.move(files, os.path.join(TEST_PATH, file_new_name))

def trading_action(data, index):
    """
    :param data: DataFrame
    :param index: Date Index for slicing
    :return: Folder destination as String
    """
    future_open = data[data['DateTime'].dt.date.astype(str) == index]['Open'].iloc[0]
    future_close = data[data['DateTime'].dt.date.astype(str) == index]['Open'].iloc[-1]
    if future_open < future_close:
        decision = 'LONG'
    else:
        decision = 'SHORT'
    return decision

def generate_gaf(images_data):
    index = 0
    for decision, data in images_data.items():
        for image_data in data:
            to_plot = [ttg.create_gaf(x)['gadf'] for x in image_data]
            try:
                ttg.create_images(X_plots=to_plot,
                              image_name='{0}'.format(index),
                              destination=decision)
            except:
                print([image_data[0], image_data[1], image_data[2], image_data[3]])
            index += 1


if __name__ == "__main__":
    pool = Pool(4)
    print(dt.datetime.now())
    print('CONVERTING TIME-SERIES TO IMAGES')
    pool.apply(data_to_image_preprocess)
    print('DONE!')
    print(dt.datetime.now())