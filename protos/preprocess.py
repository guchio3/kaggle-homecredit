import pandas as pd
import numpy as np

from tqdm import tqdm
from logging import getLogger

from data_processing.spec_analyzer \
    import HomeCreditAnalyzer
from data_processing.spec_preprocessor \
    import HomeCreditPreprocessor
from data_processing.data_io import DataIO
from utils.my_logging import logInit


def main():
    logger = getLogger(__name__)
    logInit(logger)
    logger.info('start')

    dataio = DataIO(logger=logger)
    analyzer = HomeCreditAnalyzer(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

    dfs_dict = dataio.read_csvs({
        'train': '../inputs/application_train.csv',
        'test': '../inputs/application_test.csv'})
    for df_key in dfs_dict:
        logger.info('imputing {}...'.format(df_key))
        cat_list, dis_num_list, num_list = \
            prep.feature_type_split(dfs_dict[df_key])
        dfs_dict[df_key] = prep.impute(
            dfs_dict[df_key], cat_list, strategy='most_frequent')
        dfs_dict[df_key] = prep.impute(
            dfs_dict[df_key], dis_num_list, strategy='most_frequent')
        dfs_dict[df_key] = prep.impute(
            dfs_dict[df_key], num_list, strategy='median')
        logger.info(analyzer.get_null_stat(
            dfs_dict[df_key], sort_target='null_count').head(3))
    dataio.save_csv(dfs_dict['train'], '../inputs/my_train.csv')
    dataio.save_csv(dfs_dict['test'], '../inputs/my_test.csv')

    logger.info('end')


if __name__ == '__main__':
    main()
