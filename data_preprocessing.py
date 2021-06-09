import pandas as pd
import glob
from typing import *


# Chunks DataFrames in a way that part of the data points is found in the previous chunk
def chunker(seq: pd.DataFrame, size: int, loops: int) -> Generator:
    """
    :param seq: As DataFrame
    :param size: As Integer
    :param loops: As integer
    :return: Generator with overlapping index DataFrames
    """
    rem = (seq.shape[0] - size)
    rem_split = rem // loops
    for i in range(10):
        yield seq.iloc[(i * rem_split): -(rem - (i * rem_split))]


def ensemble_data(networks_chunks: int, path: str) -> List[pd.DataFrame]:
    """
    :param networks_chunks: As Integer
    :param path: As String
    :return: List of overlapping index DataFrames
    """
    dataframes = []
    for sub_folder in ['LONG', 'SHORT']:
        images = glob.glob(path + '/{}/*.png'.format(sub_folder))  # Get path to images
        dates = [dt.split('/')[-1].split('\\')[-1].split('.')[0].replace('_', '-') for dt in images]
        data_slice = pd.DataFrame({'Images': images, 'Labels': [sub_folder] * len(images), 'Dates': dates})
        data_slice['Dates'] = pd.to_datetime(data_slice['Dates'])
        dataframes.append(data_slice)
    data = pd.concat(dataframes)
    data.sort_values(by='Dates', inplace=True)
    del data['Dates']
    shape = (data.shape[0] // 5) * 4
    loops = networks_chunks
    return list(chunker(data, shape, loops))
