import pandas as pd
import numpy as np
import scipy.stats as st


class _DescriptiveAnalysis:

    @staticmethod
    def feature_description(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = {
            'Feature': [], 'Data_Type': [], 'Initial Rows (With Nulls)': [], 'Non Null Rows': [], 'Null Percent': [], 'Unique Values': []
        }

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            db_stats['Feature'].append(feature)
            db_stats['Data_Type'].append(data[feature].dtype)
            db_stats['Initial Rows (With Nulls)'].append(data[feature].shape[0])
            db_stats['Non Null Rows'].append(data[feature].dropna().shape[0])
            db_stats['Null Percent'].append(round((data[feature].isna().sum() / data[feature].shape[0]) * 100, 2))
            db_stats['Unique Values'].append(data[feature].nunique())

        db_stats = pd.DataFrame(db_stats)

        return db_stats


    @staticmethod
    def five_point_summary(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = {
            'Feature': [], 'Mean': [], 'Median': [], 'Mode': [], 'Minimum': [], 'Maximum': [],
        }

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            db_stats['Feature'].append(feature)

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                db_stats['Mean'].append(round(np.mean(data_vals), 2))
                db_stats['Median'].append(round(np.median(data_vals)))
                db_stats['Mode'].append(st.mode(data_vals).mode[0])
                db_stats['Minimum'].append(np.min(data_vals))
                db_stats['Maximum'].append(np.max(data_vals))

            else:
                db_stats['Mean'].append(np.nan)
                db_stats['Median'].append(np.nan)
                db_stats['Mode'].append(np.nan)
                db_stats['Minimum'].append(np.nan)
                db_stats['Maximum'].append(np.nan)

        db_stats = pd.DataFrame(db_stats)

        return db_stats


    @staticmethod
    def deviation_analysis(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = {
            'Feature': [], 'Standard Deviation': [], 'Skewness': [], 'Kurtosis': []
        }

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            db_stats['Feature'].append(feature)

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                db_stats['Standard Deviation'].append(data_vals.std())
                db_stats['Skewness'].append(st.skew(data_vals, bias=False))
                db_stats['Kurtosis'].append(st.kurtosis(data_vals, bias=False))

            else:
                db_stats['Standard Deviation'].append(np.nan)
                db_stats['Skewness'].append(np.nan)
                db_stats['Kurtosis'].append(np.nan)

        db_stats = pd.DataFrame(db_stats)

        return db_stats


    @staticmethod
    def outlier_analysis(data: pd.DataFrame) -> pd.DataFrame:
        db_stats = {
            'Feature': [], '25%': [], '75%': [], '99%': [], 'IQR': [], 'UL_IQR': [], 'LL_IQR': [], 'Number_Of_Outliers_IQR': []
        }

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            db_stats['Feature'].append(feature)

            if data_vals.dtype == np.float64 or data_vals.dtype == np.int64:
                q25, q75, q99 = np.percentile(data_vals, [25, 75, 99])

                db_stats['25%'].append(q25)
                db_stats['75%'].append(q75)
                db_stats['99%'].append(q99)

                iqr = q75 - q25
                lower_lim, upper_lim = q25 - 1.5 * iqr, q75 + 1.5 * iqr

                db_stats['IQR'].append(iqr)
                db_stats['UL_IQR'].append(upper_lim)
                db_stats['LL_IQR'].append(lower_lim)

                db_stats['Number_Of_Outliers_IQR'].append(data[~((data[feature] >= lower_lim) & (data[feature] <= upper_lim))].shape[0])

            else:
                db_stats['25%'].append(np.nan)
                db_stats['75%'].append(np.nan)
                db_stats['99%'].append(np.nan)
                db_stats['IQR'].append(np.nan)
                db_stats['UL_IQR'].append(np.nan)
                db_stats['LL_IQR'].append(np.nan)
                db_stats['Number_Of_Outliers_IQR'].append(np.nan)

        db_stats = pd.DataFrame(db_stats)

        return db_stats


    @staticmethod
    def limits_analysis(data: pd.DataFrame, limits: dict) -> pd.DataFrame:
        db_stats = {
            'Feature': [], 'UL_Actual': [], 'LL_Actual': [], 'Number_Of_Outliers_Actual': []
        }

        for feature in data.columns:
            if (data[feature].dropna().shape[0] == 0) or (data[feature].dropna().nunique() <= 1):
                print(f'Warning: {feature} contains only Nulls or one unique value!')
                continue

            data_vals = np.array(data[feature].dropna())

            db_stats['Feature'].append(feature)

            if feature in limits.keys():
                lower_lim, upper_lim = limits[feature]['lower_lim'], limits[feature]['upper_lim']

                db_stats['UL_Actual'].append(upper_lim)
                db_stats['LL_Actual'].append(lower_lim)
                db_stats['Number_Of_Outliers_Actual'].append(data[~((data[feature] >= lower_lim) & (data[feature] <= upper_lim))].shape[0])

            else:
                db_stats['UL_Actual'].append(np.nan)
                db_stats['LL_Actual'].append(np.nan)
                db_stats['Number_Of_Outliers_Actual'].append(np.nan)

        db_stats = pd.DataFrame(db_stats)

        return db_stats