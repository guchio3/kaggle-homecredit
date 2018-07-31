import pandas as pd
import numpy as np

from tqdm import tqdm
from logging import getLogger
from collections import OrderedDict

from data_processing.spec_analyzer \
    import HomeCreditAnalyzer
from data_processing.spec_preprocessor \
    import HomeCreditPreprocessorAdditional
from data_processing.data_io import DataIO
from utils.my_logging import logInit

was_null_list = []


def main():
    logger = getLogger(__name__)
    logInit(logger)
    logger.info('start')

    dataio = DataIO(logger=logger)

    prep = HomeCreditPreprocessorAdditional(logger=logger)
    logger.info('loading dfs...')
    train_df = pd.read_csv('../inputs/application_train.csv')
    test_df = pd.read_csv('../inputs/application_test.csv')
#    pos_df = pd.read_csv('../inputs/POS_CASH_balance.csv')
    ins_df = pd.read_csv('../inputs/installments_payments.csv')
#    cred_df = pd.read_csv('../inputs/credit_card_balance.csv')
#    bureau_df = pd.read_csv('../inputs/bureau.csv')
#    bb_df = pd.read_csv('../inputs/bureau_balance.csv')
    prev_df = pd.read_csv('../inputs/previous_application.csv')
    train_and_test_df = pd.concat([train_df, test_df])
    train_and_test_df = pd.DataFrame(train_and_test_df[['SK_ID_CURR']])

#    logger.info('fe for bureau...')
#    bureau_df = prep.fe_bureau_and_balance(bureau_df, bb_df)
#
#    logger.info('fe for pos_cash...')
#    pos_df_curr, pos_df_prev = prep.fe_pos_cash(pos_df)
#
    logger.info('fe for instalment...')
    ins_df_curr, ins_df_prev = prep.fe_installments_payments(ins_df)
#
#    logger.info('fe for creditcard...')
#    cred_df_curr, cred_df_prev = prep.fe_credit_card_balance(cred_df)

#    prev_df = prev_df.merge(
#            pos_df_prev, on='SK_ID_PREV', how='left')
    prev_df = prev_df.merge(
            ins_df_prev, on='SK_ID_PREV', how='left')
#    prev_df = prev_df.merge(
#            cred_df_prev, on='SK_ID_PREV', how='left')
    logger.info('fe for application_prev...')
    prev_df = prep.fe_application_prev(prev_df)

    logger.info('merge and splitting fes and train, test df...')
    train_and_test_df = train_and_test_df.merge(
            prev_df, on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(
#            pos_df_curr, on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(
#            cred_df_curr, on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(
#            bureau_df, on='SK_ID_CURR', how='left')

    logger.info('fe for application...')
    train_and_test_df = prep.fe_application(train_and_test_df)

    train_df = train_and_test_df.iloc[:train_df.shape[0]]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]

    logger.info('saving train dfs...')
    dataio.save_csv(train_df, '../inputs/my_train_all_additional.csv', index=False)
    logger.info('saving test dfs...')
    dataio.save_csv(test_df, '../inputs/my_test_all_additional.csv', index=False)

    logger.info('end')

    return train_df, test_df


if __name__ == '__main__':
    main()
