import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data.utils import get_week_range_df, to_week_range
from models.ar_model import ARModelSpecification, create_ar_models_report, create_all_ar_models_report
from models.persistence_model import create_persistence

import os
from datetime import datetime

import logging

log = logging.getLogger("models.ar_model")

# Global set-up
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")

# Global set-up
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")

GROUND_TRUTH_COLUMN = 'Disease Rate'
BASELINE_SHIFT = 1
PERSISTENCE_COL_NAME = 'Persistence'
LAGS = 20

DF = get_week_range_df('week range')
DF = create_persistence(DF, BASELINE_SHIFT, persistance_col_name=PERSISTENCE_COL_NAME)

TRAIN_INTERVALS = [
    (to_week_range(2004, 2), to_week_range(2008, 52)),
    (to_week_range(2005, 1), to_week_range(2009, 52)),
    (to_week_range(2006, 2), to_week_range(2010, 52)),
    (to_week_range(2007, 2), to_week_range(2011, 52)),
    (to_week_range(2008, 2), to_week_range(2012, 52)),
    (to_week_range(2009, 2), to_week_range(2013, 52)),
    (to_week_range(2010, 2), to_week_range(2014, 52)),
    (to_week_range(2011, 2), to_week_range(2015, 52)),
    (to_week_range(2012, 2), to_week_range(2016, 52)),
    (to_week_range(2013, 2), to_week_range(2017, 52)),
]

TEST_INTERVALS = [
    (to_week_range(2009, 1), to_week_range(2009, 52)),
    (to_week_range(2010, 1), to_week_range(2010, 52)),
    (to_week_range(2011, 2), to_week_range(2011, 52)),
    (to_week_range(2012, 2), to_week_range(2012, 52)),
    (to_week_range(2013, 2), to_week_range(2013, 52)),
    (to_week_range(2014, 2), to_week_range(2014, 52)),
    (to_week_range(2015, 2), to_week_range(2015, 52)),
    (to_week_range(2016, 2), to_week_range(2016, 52)),
    (to_week_range(2017, 2), to_week_range(2017, 52)),
    (to_week_range(2018, 2), to_week_range(2018, 52)),
]

MODEL_SPECS = [
    ARModelSpecification(order=(3, 0, 1), model_class=SARIMAX),
    ARModelSpecification((3, 0, 0), seasonal_order=(3, 0, 0, 52), model_class=SARIMAX),
    ARModelSpecification((3, 0, 0), seasonal_order=(3, 0, 3, 52), model_class=SARIMAX),
    ARModelSpecification((3, 0, 0), seasonal_order=(5, 0, 0, 52), model_class=SARIMAX),

]

if __name__ == "__main__":
    path = os.path.join("ar_runs", str(datetime.now()).replace(":", "-").replace(".", "-"))
    print(str(path))
    print('Created report folder at %s ...\n' % str(path))

    os.mkdir(path)
    print('Started creating AR models report...\n')
    ar_model_report = create_ar_models_report(data=DF, ar_model_specs=MODEL_SPECS, train_interval=TRAIN_INTERVALS[0],
                                              test_interval=TEST_INTERVALS[0],
                                              additional_model_cols=[PERSISTENCE_COL_NAME],
                                              cov_type=None, train_maxiter=3000, optimize_method='powell')
    print('Finished creating AR models report...\n')
    errors_map = ar_model_report.get_errors()
    print('Writing errors to xlsx ...\n')
    with pd.ExcelWriter(os.path.join(path, 'errors.xlsx'), engine='xlsxwriter') as writer:
        errors_map['mae'].to_excel(writer, sheet_name='MAE')
        errors_map['mape'].to_excel(writer, sheet_name='MAPE')
        errors_map['mse'].to_excel(writer, sheet_name='MSE')
        errors_map['rmse'].to_excel(writer, sheet_name='RMSE')
    print('Finished writing errors to xlsx ...\n')
    print('Saving AR models plots...\n')
    ar_model_report.plot_models(include_ground_truth=True, multi_plot=True, save=True, show=False, folder_path=path)
    print('DONE...\n')
