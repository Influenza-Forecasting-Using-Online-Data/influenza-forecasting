import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy

import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order



class ARHelper:
    def __init__(self):
        self.seasonal_order = None
        self.y_test_prediction = None
        self.test_result = None
        self.y_train_prediction = None
        self.train_result = None
        self.test_interval = None
        self.validation_interval = None
        self.train_interval = None
        self.test_data = None
        self.validate_data = None
        self.model = None
        self.model_name = None
        self.data = None
        self.endog_col_name = None
        self.train_data = None
        self.order = None

    def set_model_name(self, model_name):
            self.model_name = model_name

    def set_data(self, data, endog_col_name):
        self.data = data
        self.endog_col_name = endog_col_name

    def split_dataset_by_intervals(self, train_interval, validation_interval, test_interval, print_data=False):
        assert self.data is not None
        assert self.endog_col_name is not None

        self.train_data = self.data[self.endog_col_name][train_interval[0]:train_interval[1]]
        self.validate_data = self.data[self.endog_col_name][validation_interval[0]:validation_interval[1]]
        self.test_data = self.data[self.endog_col_name][test_interval[0]:test_interval[1]]

        self.train_interval = train_interval
        self.validation_interval = validation_interval
        self.test_interval = test_interval

        if print_data is True:
            print('Training Data 1\n-------------\n' + str(self.train_data))
            print('\n')
            print('Validate Data 2\n----------\n' + str(self.validate_data))
            print('\n')
            print('Test Data 2\n----------\n' + str(self.test_data))

    def create_model(self, order, seasonal_order=None):
        self.model = sm.tsa.statespace.SARIMAX(endog=self.train_data, order=order, seasonal_order=seasonal_order)
        sorder_str = '' if seasonal_order is None else str(seasonal_order)
        self.model_name = str(order) + sorder_str
        self.order = order
        self.seasonal_order = seasonal_order

    def select_best_hyperparams(self, maxlag, ic='bic', glob=False, trend='n', seasonal=False, period=None):
        """
        Selects a model with bet performing hyperparamters for given validation
        interval on data. If model was previously set using set_model, then this
        method will overwrite it. Will also overwrite model_name to be the name
        of the model class.
        
        Returns: ar_select_order result, NOT model.
        """
        select_result = ar_select_order(self.validate_data, maxlag=maxlag, seasonal=seasonal,
                                        period=period, glob=glob, ic=ic, old_names=False)
        return select_result

    def train_model(self):
        assert self.model is not None
        assert self.train_data is not None
        assert self.validate_data is not None
        assert self.test_data is not None

        self.train_result = self.model.fit()
        self.y_train_prediction = self.train_result.predict(self.train_interval[0], self.train_interval[1])
        return self.train_result

    def test_model(self, steps=1):
        assert self.model is not None
        assert self.train_data is not None
        assert self.validate_data is not None
        assert self.test_data is not None
        assert self.train_result is not None

        temp_model = sm.tsa.statespace.SARIMAX(endog=self.data[self.endog_col_name],
                                               order=self.order, seasonal_order=self.seasonal_order)
        # https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.fix_params.html
        with temp_model.fix_params(self.train_result.params[:-1]):
            self.test_result = temp_model.fit()
            self.y_test_prediction = self.test_result.predict(self.test_interval[0], self.test_interval[1],
                                                              dynamic=False)
        return self.test_result

    def get_train_prediction_df(self, model_name=None, steps=1):
        assert self.train_result is not None
        assert steps > 0

        if model_name is None:
            model_name = self.model_name

        train_prediction_df = self.train_data.copy(deep=True)
        train_prediction_df[model_name] = self.y_train_prediction
        return train_prediction_df

    def get_test_prediction_df(self, model_name=None, steps=1):
        assert self.test_result is not None
        assert steps > 0

        if model_name is None:
            model_name = self.model_name

        test_prediction_df = self.test_data.copy(deep=True)
        test_prediction_df[model_name] = self.y_test_prediction
        return test_prediction_df


def plot_ar_results(dataset_type, arHelpers, x_format='month',figsize=(30, 10), ticks_fontsize=16,
                    legend_fontsize=20, title='', title_fontsize=20, label_fontsize=22,
                    xlim=None, ground_truth_col='Disease Rate', include_ground_truth=True):
    if isinstance(arHelpers, list) is False:
        raise Exception("arHelper object must be passed as list")
    if len(arHelpers) == 0:
        raise Exception("arHelper list is empty")

    data_interval = None
    model_result_attr = None
    interval_attr = ''
    data = None
    x_axis_locator = None
    x_axis_formatter = None
    if x_format == 'month':
        x_axis_locator = mdates.MonthLocator(bymonthday=1)
        x_axis_formatter = mdates.DateFormatter('%Y %b')
    elif x_format == 'week':
        x_axis_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        x_axis_formatter = mdates.ConciseDateFormatter(x_axis_locator)
    else:
        raise Exception("x_format must be 'month' or 'week'")

    if dataset_type == 'train':
        data_interval = arHelpers[0].train_interval
        interval_attr = 'train_interval'
        model_result_attr = 'train_result'
        data = arHelpers[0].train_data
    elif dataset_type == 'test':
        data_interval = arHelpers[0].test_interval
        interval_attr = 'test_interval'
        model_result_attr = 'test_result'
        data = arHelpers[0].test_data
    else:
        raise Exception("dataset_type must be 'train' or 'test'")

    fig, ax = plt.subplots(figsize=figsize)
    index = 0
    full_dataframe = arHelpers[0].data.copy(deep=True)[data_interval[0]:data_interval[1]]
    for arHelper in arHelpers:
        if getattr(arHelper, model_result_attr) is None:
            plt.clf()
            raise Exception("arHelper at index=" + str(index) + " missing" + str(model_result_attr))
        if getattr(arHelper, interval_attr) != data_interval:
            plt.clf()
            raise Exception("All arHelpers must have the same test interval, expected "
                            + str(data_interval) + " got " + str(arHelper.test_interval)
                            + " at arHelper index=" + str(index))
        index += 1
        if dataset_type == 'train':
            # if data.equals(arHelper.train_data) is False:
            #     plt.clf()
            #     raise Exception("train data not equal at arHelper with index=" + str(index))
            full_dataframe[str(index) + " " + str(arHelper.model_name)] = arHelper.y_train_prediction
        else:
            # if data.equals(arHelper.test_data) is False:
            #     plt.clf()
            #     raise Exception("test data not equal at arHelper with index=" + str(index))
            full_dataframe[str(index) + " " + str(arHelper.model_name)] = arHelper.y_test_prediction

    index = 0
    for arHelper in arHelpers:
        index += 1
        full_dataframe.plot(use_index=True, y=str(index) + " " + str(arHelper.model_name), x_compat=True, ax=ax)
    if include_ground_truth:
        full_dataframe.plot(use_index=True, y=ground_truth_col, x_compat=True, ax=ax)

    ax.xaxis.set_major_locator(x_axis_locator)
    ax.xaxis.set_major_formatter(x_axis_formatter)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    ax.set_xlabel(str(data_interval[0]) + ' - ' + str(data_interval[1]), fontsize=label_fontsize)
    ax.set_ylabel('ILI Rate', fontsize=label_fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)

    if xlim is not None and len(xlim) == 2:
        plt.xlim(xlim[0], xlim[1])

    plt.show()
    return full_dataframe
