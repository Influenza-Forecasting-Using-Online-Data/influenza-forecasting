import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os.path

from datetime import datetime

def get_datetime_df(index_col_name='year week'):
    parent_path = my_path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(parent_path, 'year_and_week_data_frame.csv'))
    datetime_df = pd.DataFrame({index_col_name: pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                                         format="%Y %U %w")})
    datetime_df = pd.concat([datetime_df, df], axis=1)
    return datetime_df
    
def get_week_range_df(index_col_name):
    df = get_datetime_df(index_col_name)
    df.drop(df.columns[1], axis=1, inplace=True)
    df.set_index(index_col_name, inplace=True)
    df.index = pd.DatetimeIndex(df.index, closed='left').to_period('W')
    return df
    
def to_week_range(year_num, week_num):
    return datetime.strptime(str(year_num) + " " + str(week_num) + " 1","%Y %U %w")

    