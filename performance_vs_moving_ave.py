import pandas as pd
import numpy as np

PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\TimeSeries\\'


def run_moving_ave(df):
    data = df
    data['move_ave_20'] = data['Open'].rolling(window=20).mean().fillna(method='bfill')
    data['move_ave_50'] = data['Open'].rolling(window=50).mean().fillna(method='bfill')
    return data


def buy_hold_sell(df):
    data = df
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    for i in data.index:
        # if sma30 > sma100  then buy else sell
        if data['move_ave_20'][i] > data['move_ave_50'][i]:
            if flag != 1:
                sigPriceBuy.append('BUY')  #data['Open'][i])
                sigPriceSell.append(np.nan)
                flag = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
            # print('Buy')
        elif data['move_ave_20'][i] < data['move_ave_50'][i]:
            if flag != 0:
                sigPriceSell.append('SELL')  #data['Open'][i])
                sigPriceBuy.append(np.nan)
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
            # print('sell')
        else:  # Handling nan values
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    data['Buy_Signal_Price'] = sigPriceBuy
    data['Sell_Signal_Price'] = sigPriceSell
    data['Action'] = data['Buy_Signal_Price'].fillna(data['Sell_Signal_Price'])
    data['Action'] = data['Action'].fillna('HOLD')
    del data['Buy_Signal_Price']
    del data['Sell_Signal_Price']
    return data


def check_bhs(df):
    pass


def check_cnn(df):
    pass


def produce_report(df):
    pass


if __name__ == "__main__":
    ive_data = 'IVE_tickbidask.txt'
    col_name = ['Date', 'Time', 'Open', 'High', 'Low', 'Volume']
    df = pd.read_csv(PATH + ive_data, names=col_name, header=None)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    group_dt = df.groupby(pd.Grouper(key='DateTime', freq='2h')).sum().reset_index()
    group_dt['Open'] = group_dt['Open'].replace(to_replace=0, method='ffill')
    data = group_dt[['DateTime', 'Open']].tail(10000)
    data = run_moving_ave(data)
    data = buy_hold_sell(data)