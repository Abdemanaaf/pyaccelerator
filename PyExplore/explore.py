from functools import reduce

import pandas as pd

from src.descriptive_analysis import _DescriptiveAnalysis
from src.univariate_analysis import _NVariateAnalysis


class ExploratoryDataAnalysis:


    @staticmethod
    def run_univariate(data: pd.DataFrame, column_list: list = None, save_path: str = False, **context: dict) -> None:
        if column_list is not None:
            _NVariateAnalysis.create_univariate_graphs(data=data[column_list], save_path=save_path, **context)

        else:
            _NVariateAnalysis.create_univariate_graphs(data=data, save_path=save_path, **context)


    @staticmethod
    def run_bivariate() -> None:
        pass


    @staticmethod
    def run_multivariate() -> None:
        pass


    @staticmethod
    def run_descriptive(data: pd.DataFrame, force_column: list = None, limits: dict = None, save_path: str = False) -> None:
        db_stats = pd.DataFrame()

        if force_column is not None:
            for column in force_column:
                data[column] = pd.to_numeric(data[column], errors='coerce')

        fd = _DescriptiveAnalysis.feature_description(data=data)
        fps = _DescriptiveAnalysis.five_point_summary(data=data)
        da = _DescriptiveAnalysis.deviation_analysis(data=data)
        oa = _DescriptiveAnalysis.outlier_analysis(data=data)

        if limits is not None:
            la = _DescriptiveAnalysis.limits_analysis(data=data, limits=limits)
            db_stats = reduce(lambda left, right: pd.merge(left=left, right=right, on='Feature', how='left'), [fd, fps, da, oa, la])

        else:
            db_stats = reduce(lambda left, right: pd.merge(left=left, right=right, on='Feature', how='left'), [fd, fps, da, oa])

        if save_path:
            db_stats.to_csv(save_path, index=False)

        print(db_stats)

        return db_stats


    @staticmethod
    def run_correlations() -> None:
        pass