import numpy as np
import pandas as pd

import scipy.stats as st
from sklearn.feature_selection import mutual_info_classif


class GetCorrelations:

    @staticmethod
    def point_biserial_df(X1: pd.DataFrame, X2: np.array, save_path: str = False) -> pd.DataFrame:

        if isinstance(X2, pd.DataFrame):
            raise ValueError(f'X2 should not be pd.DataFrame, found {type(X2)}')

        if (len(X2.shape)) > 1 and (X2.shape[1] != 1):
            raise ValueError(f'Expected a 1-D array X2, found shape {X2.shape}')
        
        assert X1.shape[0] == X2.shape[0], 'Unequal length arrays'

        corr = []
        
        for col in X1.columns:
            pb_out = st.pointbiserialr(X1[col].values, X2)
            corr.append([col, abs(pb_out.correlation), pb_out.correlation, pb_out.pvalue])

        corr = pd.DataFrame(data=corr, columns=['Feature', 'Point Biserial Abs Corr', 'Point Biserial Corr', 'P-Value'])
        
        if save_path:
            corr.to_csv(save_path, index=False)

        return corr
    
    @staticmethod
    def spearman_corr_df(X1: pd.DataFrame, X2: pd.DataFrame, 
        save_path: str = False, series_name: str = 'series') -> pd.DataFrame:

        assert X1.shape[0] == X2.shape[0], 'Unequal length arrays'
        
        corr = []

        for col_i in X1.columns:
            if isinstance(X2, pd.DataFrame):
                for col_j in X2.columns:
                    corr_coef, p_value = st.spearmanr(X1[col_i].values, X2[col_j].values)
                    corr.append([col_i, col_j, abs(corr_coef), corr_coef, p_value])
            else:
                corr_coef, p_value = st.spearmanr(X1[col_i].values, X2)
                corr.append([col_i, series_name, abs(corr_coef), corr_coef, p_value])

        corr = pd.DataFrame(data=corr, columns = ['Feature 1', 'Feature 2', 'Spearman Abs Corr', 'Spearman Corr', 'P-Value'])

        if save_path:
            corr.to_csv(save_path, index=False)

        return corr
    
    @staticmethod
    def pearson_corr_df(X1: pd.DataFrame, X2: pd.DataFrame, 
        save_path: str = False, series_name: str = 'series') -> pd.DataFrame:

        assert X1.shape[0] == X2.shape[0], 'Unequal length arrays'
        
        corr = []

        for col_i in X1.columns:
            if isinstance(X2, pd.DataFrame):
                for col_j in X2.columns:
                    corr_coef, p_value = st.pearsonr(X1[col_i].values, X2[col_j].values)
                    corr.append([col_i, col_j, abs(corr_coef), corr_coef, p_value])
            else:
                corr_coef, p_value = st.pearsonr(X1[col_i].values, X2)
                corr.append([col_i, series_name, abs(corr_coef), corr_coef, p_value])

        corr = pd.DataFrame(data = corr, columns=['Feature 1', 'Feature 2', 'Pearson Corr Abs', 'Pearson Corr', 'P-Value'])

        if save_path:
            corr.to_csv(save_path, index=False)

        return corr
    
    @staticmethod
    def mutual_info_gain_df(X1: pd.DataFrame, X2: np.array, save_path: str = False) -> pd.DataFrame:

        assert X1.shape[0] == X2.shape[0], 'Unequal length arrays'

        if (len(X2.shape)) > 1 and (X2.shape[1] != 1):
            raise ValueError(f'Expected a 1-D array X2, found shape {X2.shape}')

        data_cols = X1.columns
        
        mutual_info = mutual_info_classif(X1.values, X2)
        mutual_info = pd.DataFrame({'Feature': data_cols, 'MI Values': mutual_info})
        mutual_info = mutual_info.sort_values(by=['MI Values'], ascending=False)

        if save_path:
            mutual_info.to_csv(save_path, index=False)

        return mutual_info
    
    @staticmethod
    def run_corelation(X1: pd.DataFrame, X2: pd.DataFrame, correlation: str, save_path: str = False, **kwargs) -> None:

        if correlation == 'point_bi':
            corr = GetCorrelations.point_biserial_df(X1=X1, X2=X2, save_path=save_path)
        
        elif correlation == 'spearman':
            corr = GetCorrelations.spearman_corr_df(X1=X1, X2=X2, save_path=save_path, **kwargs)

        elif correlation == 'pearson':
            corr = GetCorrelations.pearson_corr_df(X1=X1, X2=X2, save_path=save_path, **kwargs)
        
        elif correlation == 'mutual_info_gain':
            corr = GetCorrelations.mutual_info_gain_df(X1=X1, X2=X2, save_path=save_path)
        else:
            raise ValueError(f"Invalid string for correlation. Valid strings are ['point_bi', 'spearman', 'pearson' and 'mutual_info_gain']")
        
        return corr


if __name__ == '__main__':
    data = pd.read_csv('./data/winequality_red.csv').dropna()
    X = data._get_numeric_data()
    y = data['quality']

    corr = GetCorrelations.run_corelation(X, y, 'spearman', series_name='series')
    print(corr)
