import pandas as pd
import numpy as np

from tqdm import tqdm
from logging import getLogger
from collections import OrderedDict

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
#    dfs_dict = dataio.read_csvs(OrderedDict({
#        'train': '../inputs/application_train.csv',
#        'test': '../inputs/application_test.csv',
#        'prev_app': '../inputs/previous_application.csv',
#        'bureau': '../inputs/bureau.csv',
#        'bureau_balance': '../inputs/bureau_balance.csv',
#        'instrallments': '../inputs/installments_payments.csv',
#        'credit': '../inputs/credit_card_balance.csv',
#        'pos': '../inputs/POS_CASH_balance.csv',
#        }))

#    analyzer = HomeCreditAnalyzer(logger=logger)
#    prep = HomeCreditPreprocessor(
#            train_df=dfs_dict['train'],
#            test_df=dfs_dict['test'],
            #prev_app_df=dfs_dict['prev_app'],
#            bureau_df=dfs_dict['bureau'],
#            bureau_balance_df=dfs_dict['bureau_balance'],
#            instrallments_df=dfs_dict['instrallments'],
#            credit_df=dfs_dict['credit'],
#            pos_df=dfs_dict['pos'],
#            logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)
    logger.info('loading dfs...')
    train_df = pd.read_csv('../inputs/application_train.csv')
    test_df = pd.read_csv('../inputs/application_test.csv')
    prev_df = pd.read_csv('../inputs/previous_application.csv')
    bureau_df = pd.read_csv('../inputs/bureau.csv')
    bb_df = pd.read_csv('../inputs/bureau_balance.csv')
    pos_df = pd.read_csv('../inputs/POS_CASH_balance.csv')
    ins_df = pd.read_csv('../inputs/installments_payments.csv')
    cred_df = pd.read_csv('../inputs/credit_card_balance.csv')
    train_and_test_df = pd.concat([train_df, test_df])

    logger.info('fe for previous application...')
    train_and_test_df = prep.fe_application(train_and_test_df)
    # add the info whether the features were null or not
#    train_and_test_df = prep.add_was_null(
#            train_and_test_df, null_rat_th=0.1,
#            special_list=[
#                'EXT_SOURCE_1',
#                'EXT_SOURCE_2',
#                'EXT_SOURCE_3',
#                'PAYMENT_RATE',
#                'DAYS_BIRTH',
#                'DAYS_EMPLOYED',
#                'ANNUITY_INCOME_PERC',
#                'ACTIVE_RATE_CREDIT_MAX',
#                ])
#    train_and_test_df = prep.auto_impute(train_and_test_df)
    prev_df = prep.fe_application_prev(prev_df)
    prev_df = prep.add_was_null(prev_df, special_list=[
        'APPROVED_CNT_PAYMENT_MEAN',
        'PREV_CNT_PAYMENT_MEAN',
        'PREV_APP_CREDIT_PERC_MEAN',
        ])
#    prev_df = prep.auto_impute(prev_df)
    bureau_df = prep.fe_bureau_and_balance(bureau_df, bb_df)
    bureau_df = prep.add_was_null(bureau_df, special_list=[
        'ACTIVR_DAYS_CREDIT_MEAN',
        'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
        'BURO_DAYS_CREDIT_ENDDATE_MEAN',
        'BURO_AMT_CREDIT_SUM_MEAN',
        'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
        'BURO_DAYS_CREDIT_MEAN',
        'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
        ])
#    bureau_df = prep.auto_impute(bureau_df)
    pos_df = prep.fe_pos_cash(pos_df)
    pos_df = prep.add_was_null(pos_df, special_list=[
        'POS_MONTHS_BALANCE_SIZE',
        ])
#    pos_df = prep.auto_impute(pos_df)
    ins_df = prep.fe_installments_payments(ins_df)
    ins_df = prep.add_was_null(ins_df, special_list=[
        'INSTAL_DPD_MEAN',
        'INSTAL_AMT_PAYMENT_MEAN',
        'INSTAL_DBD_SUM',
        'INSTAL_AMT_PAYMENT_MIN',
        'INSTAL_PAYMENT_DIFF_MEAN',
        ])
#    ins_df = prep.auto_impute(ins_df)
    cred_df = prep.fe_credit_card_balance(cred_df)
#    cred_df = prep.add_was_null(cred_df)
#    cred_df = prep.auto_impute(cred_df)

    logger.info('merge and splitting fes and train, test df...')
    train_and_test_df = train_and_test_df.merge(
            prev_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            bureau_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            pos_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            ins_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            cred_df, on='SK_ID_CURR', how='left')

#    train_and_test_df = prep.auto_impute(
#            train_and_test_df, mode='min')
    train_df = train_and_test_df.iloc[:train_df.shape[0]]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]

#    prep.impute_all()
#    prep.add_prev_loan_cnt()

    logger.info('saving train and test dfs...')
    dataio.save_csv(train_df, '../inputs/my_train_2_w_missing_and_was_null_with_prev5.csv', index=False)
    dataio.save_csv(test_df, '../inputs/my_test_2_w_missing_and_was_null_with_prev5.csv', index=False)
#    dataio.save_csv(prep.train_df, '../inputs/my_train_2.csv', index=False)
#    dataio.save_csv(prep.test_df, '../inputs/my_test_2.csv', index=False)

    logger.info('end')


if __name__ == '__main__':
    main()
