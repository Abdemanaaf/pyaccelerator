import pandas as pd

from explore import ExploratoryDataAnalysis


def run() -> None:
    data = pd.read_csv('./data/Titanic_Classification_Dataset_Train.csv')

    ExploratoryDataAnalysis.run_descriptive(
        data, force_column=['Ticket'], limits={'Age': {'upper_lim': 50, 'lower_lim': 18}}, save_path='./results/descriptive_stats/db_stats.csv'
    )

    ExploratoryDataAnalysis.run_univariate(
        data, save_path='./results/plots/univariate/'
    )


if __name__ == '__main__':
    run()