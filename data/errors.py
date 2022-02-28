import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd



class Errors:
    # upload weekly data and insert the column names for y_actual and y_predicted
    def __init__(self, data, actual_col_name, predicted_col_name):
        # TODO: add booleans to choose what errors you want calculated
        self.data_weekly = data
        self.actual_name = actual_col_name
        self.predicted_name = predicted_col_name
        self.year_df = None

    def _get_mae(self, actual, predicted):
        self.mae = mean_absolute_error(actual, predicted)
        return self.mae

    def _get_mape(self, actual, predicted):
        self.mape = np.mean(np.abs((actual - predicted)/actual))*100
        return self.mape

    def _get_mse(self, actual, predicted):
        self.mse = mean_squared_error(actual, predicted)
        return self.mse

    # TODO: add formulas
    def _get_negative_likelihood(self, actual, predicted):
        self.neg_like = []
        return self.neg_like

    def _get_rmse(self, actual, predicted):
        self.rmse = np.sqrt(self._get_mse(actual, predicted))
        return self.rmse



    def show_errors(self):
        # df cols:  week range | year | week | MAE | MAPE | MSE | NEG_LIKELIHOOD | RMSE

        errors_df = self._get_yearly_index_df()

        # get stats per year
        (MAEs, MAPEs, MSEs, NEGs, RMSEs) = self._get_stats_per_year()

        # append
        errors_df['MAE'] = MAEs
        errors_df['MAPE'] = MAPEs
        errors_df['MSE'] = MSEs
        errors_df['NEG'] = NEGs
        errors_df['RMSE'] = RMSEs

        return errors_df

    ### HELPER FUNCTIONS

    def _extract_cols(self):
        # get columns of interest from data, to calculate values based on them
        self.actual = self.data_weekly[self.actual_name]
        self.predicted = self.data_weekly[self.predicted_name]

    def _get_yearly_index_df(self):
        yearly_df = self.data_weekly.iloc[:, 0:2]

        self.years = yearly_df['year'].unique()
        weeks=[]
        for year in self.years:
            weeks.append(len(yearly_df[yearly_df['year']==year]))

        temp_dict = {'years' : self.years,
                     'no. of weeks' : weeks}
        base_df = pd.DataFrame(temp_dict, index=None)

        return base_df

    def _get_stats_per_year(self):
        MAEs = []
        MAPEs = []
        MSEs = []
        NEGs = []
        RMSEs = []
        for year in self.years:
            self.year_df = self.data_weekly[self.data_weekly['year'] == year]
            actual = self.year_df[self.actual_name]
            predicted = self.year_df[self.predicted_name]

            mae_year = self._get_mae(actual, predicted)
            mape_year = self._get_mape(actual, predicted)
            mse_year = self._get_mse(actual, predicted)
            neg_year = self._get_negative_likelihood(actual, predicted)
            rmse_year = self._get_rmse(actual, predicted)

            MAEs.append(mae_year)
            MAPEs.append(mape_year)
            MSEs.append(mse_year)
            NEGs.append(neg_year)
            RMSEs.append(rmse_year)

        return (MAEs, MAPEs, MSEs, NEGs, RMSEs)
