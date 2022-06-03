import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


_DEFAULT_PARAMS_DISTPLOT = {
    'bins': 5, 'kde': True, 'color': 'teal', 'kde_kws': dict(linewidth=4, color='black'), 'rug': True
}


class _NVariateAnalysis:

    @staticmethod
    def create_univariate_graphs(data: pd.DataFrame, save_path: str = False, context: dict = None, categorical_columns: list = None):
        numeric_cols = data._get_numeric_data().columns

        if categorical_columns is not None:
            for feature in categorical_columns:
                fig_dist, axes_dist = plt.subplots(1, 1, figsize=(30, 15))
                sb.set_style('whitegrid')

                sb.countplot(x=data[feature].dropna(), ax=axes_dist[0]).set(title='Full Dataset', xlabel=f'{feature.upper()}')

                if save_path:
                    fig_dist.savefig(f'{save_path}{feature}.png')


            for feature in numeric_cols:
                if feature not in categorical_columns:
                    fig_dist, axes_dist = plt.subplots(2, 1, figsize=(30, 15))
                    sb.set_style('whitegrid')

                    if context is not None:
                        sb.distplot(x=data[feature].dropna(), **context, ax=axes_dist[0]).set(title='Full Dataset', xlabel=f'{feature.upper()}')
                    else:
                        sb.distplot(x=data[feature].dropna(), **_DEFAULT_PARAMS_DISTPLOT, ax=axes_dist[0]).set(title='Full Dataset', xlabel=f'{feature.upper()}')

                    sb.boxplot(x=data[feature].dropna(), ax=axes_dist[1]).set(xlabel=f'{feature.upper()}')

                    if save_path:
                        fig_dist.savefig(f'{save_path}{feature}.png')

        else:
            for feature in numeric_cols:
                fig_dist, axes_dist = plt.subplots(2, 1, figsize=(30, 15))
                sb.set_style('whitegrid')

                if context is not None:
                    sb.distplot(x=data[feature].dropna(), **context, ax=axes_dist[0]).set(title='Full Dataset', xlabel=f'{feature.upper()}')
                else:
                    sb.distplot(x=data[feature].dropna(), **_DEFAULT_PARAMS_DISTPLOT, ax=axes_dist[0]).set(title='Full Dataset', xlabel=f'{feature.upper()}')

                sb.boxplot(x=data[feature].dropna(), ax=axes_dist[1]).set(xlabel=f'{feature.upper()}')

                if save_path:
                    fig_dist.savefig(f'{save_path}{feature}.png')


    # @staticmethod
    # def create_bivariate_graphs(data: pd.DataFrame, feat: str, label_col: str, path: str):
    #     uniq_label_list = list(data[label_col].unique())

    #     fig_dist, axes_dist = plt.subplots(3, len(uniq_label_list), figsize=(30, 15))
    #     sb.set_style('whitegrid')

    #     for label in range(len(uniq_label_list)):
    #         X_data = data[data[label_col] == uniq_label_list[label]]

    #         sb.distplot(
    #             x=X_data[feat], bins=25, kde=True, color='teal', kde_kws=dict(linewidth=4, color='black'),
    #             rug=True, ax=axes_dist[0, label]
    #         ).set(title=f'{uniq_label_list[label]}'.upper())

    #         sb.boxplot(X_data[feat], ax=axes_dist[1, label]).set(title=f'{uniq_label_list[label]}'.upper())

    #         sb.stripplot(
    #             y=X_data[feat].values, x=X_data[label_col].values, ax=axes_dist[2, label]
    #         ).set(title=f'{uniq_label_list[label]}'.upper())

    #     fig_dist.savefig(f'{path}/{feat}.png')