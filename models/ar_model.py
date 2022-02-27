import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os.path

from datetime import datetime

from statsmodels.tsa.ar_model import ar_select_order
import copy

class ARHelper:
    def set_model(self, model):
        self.model = model
    
    def set_data(self, data, output_col_name):
        self.data = data
        self.output_col_name = output_col_name
        
    def split_dataset_by_intervals(self, train_interval, validation_interval, test_interval):
        self.train_data = data[output_col_name][train_interval[0]:train_interval[1]]
        self.validate_data = data[validation_interval[0]:validation_interval[1]]
        self.test_data = data[test_interval[0]:test_interval[1]]
        
        self.train_interval = train_interval
        self.validation_interval = validation_interval
        self.test_interval = test_interval
    
    def select_best_hyperparams(maxlag, ic='bic', glob=False, trend='n', seasonal=False, period=None):
        """
        Selects a model with bet performing hyperparamters for given validation
        interval on data. If model was previously set using set_model, then this
        method will overwrite it.
        
        Returns: ar_select_order result, NOT model.
        """
        select_result = ar_select_order(self.validate_data, maxlag=maxlag, seasonal=seasonal, 
                                        period=period, glob=glob, ic=ic)
        self.model = select_result.model
        return select_result
        
    def train_model(self):
        assert self.model is not None
        assert self.train_data is not None
        assert self.validate_data is not None
        assert self.test_data is not None

        self.train_result = model.fit()
        return self.train_result.copy(deep=True)
    
    def test_model(self, steps=1):
        assert self.model is not None
        assert self.train_data is not None
        assert self.validate_data is not None
        assert self.test_data is not None
        assert self.train_result is not None

        # Prepare model for testing using learned coefficients
        temp_model = copy.deepcopy(self.model)
        temp_model.endog = test_data
        
        self.test_result = self.temp_model.filter(self.train_result.params)
        return self.test_result.copy(deep=True)
    
    def get_train_prediction_df(model_name='AR', steps=1):
        assert self.train_result is not None
        assert steps > 0
        
        y_train_prediction = self.train_result.predict(train_interval[0], train_interval[1])
        
        train_prediction_df = self.train_data.copy(deep=True)
        train_prediction_df[model_name] = y_train_prediction
        return train_prediction_df
    
    def get_test_prediction_df(model_name='AR', steps=1):
        assert self.test_result is not None
        assert steps > 0
        
        y_test_prediction = self.train_result.predict(self.train_interval[0], self.train_interval[1])
        
        test_prediction_df = self.test_data.copy(deep=True)
        test_prediction_df[model_name] = y_test_prediction
        return test_prediction_df
    
    def get_mae_df():
        pass
    
    def get_mape_df():
        pass
    
    def get_rmse_df():
        pass