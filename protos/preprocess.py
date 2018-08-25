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


#was_null_list = pd.read_csv('../importances/importance_2018-07-16-16-51-06.csv')[:100]
#was_null_list = pd.read_csv('../importances/lgbm_importances01.csv')[:100]
was_null_list = []
latest_month = 12


def main():
    logger = getLogger(__name__)
    logInit(logger)
    logger.info('start')

    dataio = DataIO(logger=logger)

    prep = HomeCreditPreprocessor(logger=logger)
    logger.info('loading dfs...')
    train_df = pd.read_csv('../inputs/application_train.csv')
    test_df = pd.read_csv('../inputs/application_test.csv')
    pos_df = pd.read_csv('../inputs/POS_CASH_balance.csv')
    ins_df = pd.read_csv('../inputs/installments_payments.csv')
    cred_df = pd.read_csv('../inputs/credit_card_balance.csv')
    bureau_df = pd.read_csv('../inputs/bureau.csv')
    bb_df = pd.read_csv('../inputs/bureau_balance.csv')
    prev_df = pd.read_csv('../inputs/previous_application.csv')
    train_and_test_df = pd.concat([train_df, test_df])

    # add missing value info
    train_and_test_df['IS_IN_PREV'] = train_and_test_df.SK_ID_CURR.isin(set(prev_df.SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_PREV_LATEST'] = train_and_test_df.SK_ID_CURR.isin(set(prev_df[prev_df.DAYS_FIRST_DUE >= -1 * latest_month].SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_CC'] = train_and_test_df.SK_ID_CURR.isin(set(cred_df.SK_ID_CURR.tolist()))
#    train_and_test_df['IS_IN_CC_LATEST'] = train_and_test_df.SK_ID_CURR.isin(set(cred_df.SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_INSTAL'] = train_and_test_df.SK_ID_CURR.isin(set(ins_df.SK_ID_CURR.tolist()))
#    train_and_test_df['IS_IN_INSTAL_LATEST'] = train_and_test_df.SK_ID_CURR.isin(set(ins_df.SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_POS'] = train_and_test_df.SK_ID_CURR.isin(set(pos_df.SK_ID_CURR.tolist()))
#    train_and_test_df['IS_IN_POS_LATEST'] = train_and_test_df.SK_ID_CURR.isin(set(pos_df.SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_BUREAU'] = train_and_test_df.SK_ID_CURR.isin(set(bureau_df.SK_ID_CURR.tolist()))
    train_and_test_df['IS_IN_BUREAU_LATEST'] = train_and_test_df.SK_ID_CURR.isin(set(bureau_df[bureau_df.DAYS_CREDIT >= -1 * 30 * latest_month].SK_ID_CURR.tolist()))

    logger.info('fe for bureau...')
    bureau_df_latest = prep.fe_bureau_and_balance(bureau_df[bureau_df.DAYS_CREDIT >= -1 * 30 * latest_month], bb_df)
    bureau_df_latest.columns = pd.Index(['LATEST_' + e.upper() for e in bureau_df_latest.columns.tolist()])
    bureau_df_latest_2 = prep.fe_bureau_and_balance(bureau_df[bureau_df.DAYS_CREDIT >= -1 * 30 * latest_month * 3], bb_df)
    bureau_df_latest_2.columns = pd.Index(['LATEST_2_' + e.upper() for e in bureau_df_latest.columns.tolist()])
    bureau_df = prep.fe_bureau_and_balance(bureau_df, bb_df)

    logger.info('fe for pos_cash...')
    pos_df_curr_latest, pos_df_prev_latest = prep.fe_pos_cash(pos_df[pos_df.MONTHS_BALANCE >= -1 * latest_month])
    pos_df_curr_latest.columns = pd.Index(['LATEST_' + e.upper() for e in pos_df_curr_latest.columns.tolist()])
    pos_df_curr_latest_2, pos_df_prev_latest_2 = prep.fe_pos_cash(pos_df[pos_df.MONTHS_BALANCE >= -1 * latest_month * 3])
    pos_df_curr_latest_2.columns = pd.Index(['LATEST_2_' + e.upper() for e in pos_df_curr_latest.columns.tolist()])
    pos_df_curr, pos_df_prev = prep.fe_pos_cash(pos_df)

    logger.info('fe for instalment...')
    ins_df_curr_latest, ins_df_prev_latest = prep.fe_installments_payments(ins_df[ins_df.DAYS_ENTRY_PAYMENT >= -1 * 30 * latest_month])
    ins_df_curr_latest.columns = pd.Index(['LATEST_' + e.upper() for e in ins_df_curr_latest.columns.tolist()])
    ins_df_curr_latest_2, ins_df_prev_latest_2 = prep.fe_installments_payments(ins_df[ins_df.DAYS_ENTRY_PAYMENT >= -1 * 30 * latest_month * 3])
    ins_df_curr_latest_2.columns = pd.Index(['LATEST_2_' + e.upper() for e in ins_df_curr_latest.columns.tolist()])
    ins_df_curr, ins_df_prev = prep.fe_installments_payments(ins_df)

    logger.info('fe for creditcard...')
    cred_df_curr_latest, cred_df_prev_latest = prep.fe_credit_card_balance(cred_df[cred_df.MONTHS_BALANCE >= -1 * latest_month])
    cred_df_curr_latest.columns = pd.Index(['LATEST_' + e.upper() for e in cred_df_curr_latest.columns.tolist()])
    cred_df_curr_latest_2, cred_df_prev_latest_2 = prep.fe_credit_card_balance(cred_df[cred_df.MONTHS_BALANCE >= -1 * latest_month * 3])
    cred_df_curr_latest_2.columns = pd.Index(['LATEST_2_' + e.upper() for e in cred_df_curr_latest.columns.tolist()])
    cred_df_curr, cred_df_prev = prep.fe_credit_card_balance(cred_df)

#    prev_df = prep.fe_application_prev(prev_df)
    logger.info('merging to application prev...')
#    prev_df_latest = prev_df[prev_df.DAYS_FIRST_DUE >= -1 * latest_month]
#    prev_df_latest = prev_df_latest.merge(
#            pos_df_prev, on='SK_ID_PREV', how='left')
#    prev_df_latest = prev_df.merge(
#            ins_df_prev, on='SK_ID_PREV', how='left')
#    prev_df_latest = prev_df_latest.merge(
#            cred_df_prev, on='SK_ID_PREV', how='left')
    prev_df_latest = prev_df[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
    prev_df_latest_2 = prev_df[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
    prev_df_latest = prev_df_latest.merge(
            pos_df_prev_latest, on='SK_ID_PREV', how='left')
    prev_df_latest = prev_df_latest.merge(
            ins_df_prev_latest, on='SK_ID_PREV', how='left')
    prev_df_latest = prev_df_latest.merge(
            cred_df_prev_latest, on='SK_ID_PREV', how='left')
    prev_df_latest.columns = pd.Index(['LATEST_' + e.upper() if e.upper() not in ['SK_ID_PREV', 'SK_ID_CURR'] else e.upper() for e in prev_df_latest.columns.tolist()])
    prev_df_latest_2 = prev_df_latest_2.merge(
            ins_df_prev_latest_2, on='SK_ID_PREV', how='left')
    prev_df_latest_2 = prev_df_latest_2.merge(
            cred_df_prev_latest_2, on='SK_ID_PREV', how='left')
    prev_df_latest_2 = prev_df_latest_2.merge(
            pos_df_prev_latest_2, on='SK_ID_PREV', how='left')
    prev_df_latest_2.columns = pd.Index(['LATEST_2_' + e.upper() if e.upper() not in ['SK_ID_PREV', 'SK_ID_CURR'] else e.upper() for e in prev_df_latest_2.columns.tolist()])

    prev_df = prev_df.merge(
            pos_df_prev, on='SK_ID_PREV', how='left')
    prev_df = prev_df.merge(
            ins_df_prev, on='SK_ID_PREV', how='left')
    prev_df = prev_df.merge(
            cred_df_prev, on='SK_ID_PREV', how='left')

    logger.info('fe for application_prev...')
    prev_df_latest = prep.fe_application_prev_latest(prev_df_latest)
    prev_df_latest_2 = prep.fe_application_prev_latest(prev_df_latest_2)
#    prev_df_latest = prep.fe_application_prev(prev_df_latest)
#    prev_df_latest = prep.fe_application_prev_latest_2(prev_df_latest)
#    prev_df_latest.columns = pd.Index(['LATEST_' + e.upper() if e.upper() not in ['SK_ID_PREV', 'SK_ID_CURR'] else e.upper() for e in prev_df_latest.columns.tolist()])
    prev_df = prep.fe_application_prev(prev_df)
#    prev_df = prep.add_was_null(prev_df,
#            special_list=was_null_list.feature.tolist())
#    prev_df = prep.add_was_null(prev_df)
#    prev_df = prep.auto_impute(prev_df)

    logger.info('merge and splitting fes and train, test df...')
#    train_and_test_df = train_and_test_df[['TARGET', 'SK_ID_CURR']]
    train_and_test_df = train_and_test_df.merge(
            prev_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            prev_df_latest, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            prev_df_latest_2, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            pos_df_curr, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            pos_df_curr_latest, on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(
#            ins_df_curr, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            cred_df_curr, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            cred_df_curr_latest, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            cred_df_curr_latest_2, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            bureau_df, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            bureau_df_latest, on='SK_ID_CURR', how='left')
    train_and_test_df = train_and_test_df.merge(
            bureau_df_latest_2, on='SK_ID_CURR', how='left')

    logger.info('fe for application...')
    train_and_test_df = prep.fe_application(train_and_test_df)

#    train_and_test_df = prep.auto_impute(
#            train_and_test_df, mode='min')
    train_df = train_and_test_df.iloc[:train_df.shape[0] - 4].reset_index()
    test_df = train_and_test_df.iloc[train_df.shape[0]:].reset_index()

#    prep.impute_all()
#    prep.add_prev_loan_cnt()

    logger.info('saving train and test dfs...')
    logger.info('saving train ...')
    #dataio.save_csv(train_df, '../inputs/my_train_all.csv', index=False)
    train_df.to_feather('../inputs/my_train_all.fth')
    logger.info('saving test ...')
    #dataio.save_csv(test_df, '../inputs/my_test_all.csv', index=False)
    test_df.to_feather('../inputs/my_test_all.fth')

    logger.info('end')

    return train_df, test_df


if __name__ == '__main__':
    main()
