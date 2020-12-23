def frequency_format(df, frequency):
    """
    :param df: DataFrame
    :param frequency: String representing a time freguency
    :return: New DataFrame grouped by time frequency
    """
    group_dt = df.groupby(pd.Grouper(key='DateTime', freq=frequency)).sum().reset_index()
    group_dt['Open'] = group_dt['Open'].replace(to_replace=0, method='ffill')
    return group_dt
