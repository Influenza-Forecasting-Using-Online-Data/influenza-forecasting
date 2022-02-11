import pandas as pd
import numpy as np
import os.path

def get_datetime_dataframe():
    parent_path = my_path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(parent_path, 'year_and_week_data_frame.csv'))
    datetime_df = pd.DataFrame({"year week": pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                                         format="%Y %U %w")})
    return pd.concat([datetime_df, df], axis=1)