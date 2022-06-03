from sys import getsizeof
from tabulate import tabulate

import pandas as pd
import numpy as np
import scipy.stats as st


class _DescriptiveAnalysis:

    @staticmethod
    def dataset_statistics(data: pd.DataFrame) -> None:
        num_records, num_feat = data.shape

        missing_cells = data.isna().sum().sum()
        missing_percent = (missing_cells / (data.shape[0] * data.shape[1])) * 100

        duplicate_rows = data.duplicated().sum()
        duplicate_percent = (duplicate_rows / data.shape[0]) * 100

        data_size = getsizeof(data)

        num_col, cat_col = 0, 0

        for feature in data.columns: cat_col += int(data[feature].dtype == 'object')
        else: num_col += 1



    @staticmethod
    def feature_description(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = []

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            db_stats.append([
                feature, data[feature].dtype, data[feature].shape[0], data[feature].dropna().shape[0],
                round((data[feature].isna().sum() / data[feature].shape[0]) * 100, 2),
                data[feature].nunique()
            ])

        db_stats = pd.DataFrame(data=db_stats, columns=[
            'Feature', 'Data_Type', 'Initial Rows (With Nulls)', 'Non Null Rows', 'Null Percent', 'Unique Values'
        ])

        return db_stats


    @staticmethod
    def five_point_summary(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = []

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                db_stats.append([
                    feature, round(np.mean(data_vals), 2), round(np.median(data_vals)), st.mode(data_vals).mode[0], np.min(data_vals),
                    np.max(data_vals)
                ])

            else:
                db_stats.append([feature, np.nan, np.nan, np.nan, np.nan, np.nan])

        db_stats = pd.DataFrame(data=db_stats, columns=['Feature', 'Mean', 'Median', 'Mode', 'Minimum', 'Maximum'])

        return db_stats


    @staticmethod
    def deviation_analysis(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = []

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                db_stats.append([feature, data_vals.std(), st.skew(data_vals, bias=False), st.kurtosis(data_vals, bias=False)])

            else:
                db_stats.append([feature, np.nan, np.nan, np.nan])

        db_stats = pd.DataFrame(data=db_stats, columns=['Feature', 'Standard Deviation', 'Skewness', 'Kurtosis'])

        return db_stats


    @staticmethod
    def outlier_analysis(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = []

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                q25, q75, q99 = np.percentile(data_vals, [25, 75, 99])

                iqr = q75 - q25
                lower_lim, upper_lim = q25 - 1.5 * iqr, q75 + 1.5 * iqr

                db_stats.append([
                    feature, q25, q75, q99, iqr, upper_lim, lower_lim, data[~((data[feature] >= lower_lim) & (data[feature] <= upper_lim))].shape[0]
                ])

            else:
                db_stats.append([feature, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,])

        db_stats = pd.DataFrame(data=db_stats, columns=['Feature', '25%', '75%', '99%', 'IQR', 'UL_IQR', 'LL_IQR', 'Number_Of_Outliers_IQR'])

        return db_stats


    @staticmethod
    def limits_analysis(data: pd.DataFrame, limits: dict) -> pd.DataFrame:
        db_stats = []

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            if feature in limits.keys():
                lower_lim, upper_lim = limits[feature]['lower_lim'], limits[feature]['upper_lim']

                db_stats.append([
                    feature, upper_lim, lower_lim, data[~((data[feature] >= lower_lim) & (data[feature] <= upper_lim))].shape[0]
                ])

            else:
                db_stats.append([feature, np.nan, np.nan, np.nan])

        db_stats = pd.DataFrame(data=db_stats, columns=['Feature', 'UL_Actual', 'LL_Actual', 'Number_Of_Outliers_Actual'])

        return db_stats