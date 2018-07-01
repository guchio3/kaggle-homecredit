import numpy as np
import pandas as pd

from tqdm import tqdm
import gc

# sys.append('./')
from .preprocessor import Preprocessor


HEAD_SIZE = 100
SUB_HEAD_SIZE = 3


class HomeCreditPreprocessor(Preprocessor):
    def __init__(self, logger=None,
                 train_df=None, test_df=None, prev_app_df=None,
                 bureau_df=None, bureau_balance_df=None,
                 instrallments_df=None, credit_df=None, pos_df=None,
                 ):
        super(HomeCreditPreprocessor, self).__init__(logger=logger)
        self.train_df = train_df
        self.test_df = test_df
        self.prev_app_df = prev_app_df
        self.bureau_df = bureau_df
        self.bureau_balance_df = bureau_balance_df
        self.instrallments_df = instrallments_df
        self.credit_df = credit_df
        self.pos_df = pos_df
        self.dfs = [
            self.train_df,
            self.test_df,
            self.prev_app_df,
            self.bureau_df,
            self.bureau_balance_df,
            self.instrallments_df,
            self.credit_df,
            self.pos_df,
        ]

    def add_was_null(self, df, mode='auto', null_rat_th=0.1, special_list=[]):
        '''
        if you set special_list, null_rat_th is ignored.

        '''
        self.logger.info('adding was null info...')
        total_count = df.shape[0]
        if mode == 'auto':
            for col in df.columns.values:
                if len(special_list) > 0:
                    if col in special_list:
                        self.logger.info('{} + was null...'.format(col))
                        df['WAS_NAN_' + col] = \
                            df[col].isnull().astype(int)
                else:
                    null_count = df[col].isnull().sum()
                    null_ratio = null_count / total_count
                    if null_ratio > null_rat_th:
                        self.logger.info('{} + was null...'.format(col))
                        df['WAS_NAN_' + col] = \
                            df[col].isnull().astype(int)
        else:
            assert NotImplementedError
        return df

    def auto_impute(self, df, mode='normal'):
        if mode == 'normal':
            cat_list, dis_num_list, num_list = \
                self.feature_type_split(df)
            self.logger.info('Now imputing {}...'.format(cat_list))
            df = self.impute(
                #            df, cat_list, strategy='most_frequent')
                df, cat_list, strategy='fill')
            self.logger.info('Now imputing {}...'.format(dis_num_list))
            df = self.impute(
                df, dis_num_list, strategy='most_frequent')
            self.logger.info('Now imputing {}...'.format(num_list))
            df = self.impute(
                df, num_list, strategy='median')
        elif mode == 'min':
            cat_list, dis_num_list, num_list = \
                self.feature_type_split(df)
            df = self.impute(
                df, dis_num_list, strategy='min')
            self.logger.info('Now imputing {}...'.format(num_list))
            df = self.impute(
                df, num_list, strategy='min')
        return df

    def fe_application(self, df):
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        #        df = df[df['CODE_GENDER'] != 'XNA']

        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) &
                ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
            'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
        df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / \
            (1 + df['CNT_CHILDREN'])
        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
        df['NEW_EMPLOY_TO_BIRTH_RATIO'] = \
            df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / \
            (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * \
            df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1',
                                         'EXT_SOURCE_2',
                                         'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_EXT_SCORES_STD'] = df[['EXT_SOURCE_1',
                                       'EXT_SOURCE_2',
                                       'EXT_SOURCE_3']].std(axis=1)
        df['NEW_EXT_SOURCE_1M2'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
        df['NEW_EXT_SOURCE_2M3'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
        df['NEW_EXT_SOURCE_3M1'] = df['EXT_SOURCE_3'] - df['EXT_SOURCE_1']
#        df['NEW_EXT_SCORES_STD'] = df['NEW_EXT_SCORES_STD'].fillna(
#            df['NEW_SCORES_STD'].mean())
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / \
            df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = \
            df['DAYS_LAST_PHONE_CHANGE'] / \
            df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / \
            df['AMT_INCOME_TOTAL']

        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
#        # Categorical features with One-Hot encode
        dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                     'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                     'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                     'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                     'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                     'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                     'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                     'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                     'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                     'FLAG_DOCUMENT_21']
        df = df.drop(dropcolum, axis=1)
        gc.collect()
        return df

    def fe_application_prev(self, df):
        # add raw sequential information processing fe
        seq_agg = {
            'NAME_CONTRACT_STATUS': ['first']
        }
        seq_agg = df.groupby('SK_ID_CURR').agg(
            {**seq_agg})
        seq_agg.columns = pd.Index(
            ['PREV_SEQ_' + e[0] + "_" + e[1].upper()
             for e in seq_agg.columns.tolist()])
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        self.logger.info(
            'categorial features of previous application are... {}'
            .format(cat_cols))
        # Days 365.243 values -> nan
        df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        # Add feature: value ask / value received percentage
        df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'APP_CREDIT_PERC': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']
        df_agg = df.groupby('SK_ID_CURR').head(HEAD_SIZE).\
            groupby('SK_ID_CURR').\
            agg({**num_aggregations, **cat_aggregations})
        df_agg.columns = pd.Index(
            ['PREV_' + e[0] + "_" + e[1].upper()
             for e in df_agg.columns.tolist()])
        df_agg_pref = df.groupby('SK_ID_CURR').head(SUB_HEAD_SIZE).\
            groupby('SK_ID_CURR').\
            agg({**num_aggregations, **cat_aggregations})
        df_agg_pref.columns = pd.Index(
            ['PREV_PREF_' + e[0] + "_" + e[1].upper()
             for e in df_agg_pref.columns.tolist()])
        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # previous Applications: Approved Applications - only numerical features
        approved = df[df['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').head(HEAD_SIZE)\
            .groupby('SK_ID_CURR').\
            agg(num_aggregations)
#        app_agg_cols = approved_agg.columns.tolist()
        approved_agg.columns = pd.Index(
            ['APPROVED_' + e[0] + "_" + e[1].upper()
             for e in approved_agg.columns.tolist()])
        df_agg = df_agg.merge(approved_agg, how='left', on='SK_ID_CURR')
        # dfious Applications: Refused Applications - only numerical features
        refused = df[df['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').head(HEAD_SIZE)\
            .groupby('SK_ID_CURR')\
            .agg(num_aggregations)
        refused_agg.columns = pd.Index(
            ['REFUSED_' + e[0] + "_" + e[1].upper()
             for e in refused_agg.columns.tolist()])
        df_agg = df_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        df_agg['PREV_CNT'] = df.groupby('SK_ID_CURR').size()
        df_agg['PREV_REFUSED_CNT'] = refused.groupby('SK_ID_CURR').size()
        df_agg['PREV_REFUSED_RATIO'] = df_agg['PREV_CNT'] /\
            df_agg['PREV_REFUSED_CNT']
        df_agg = df_agg.merge(seq_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, df

#        for e in app_agg_cols:
#            df_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = \
#                    df_agg['APPROVED_' + e[0] + "_" + e[1].upper()] /\
#                    df_agg['REFUSED_' + e[0] + "_" + e[1].upper()]

        gc.collect()
        return df_agg

    # Preprocess bureau.csv and bureau_balance.csv
    def fe_bureau_and_balance(self, bureau, bb):
        bureau, bureau_cat = self.onehot_encoding(bureau, drop_first=False)
        bb, bb_cat = self.onehot_encoding(bb, drop_first=False)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                                   for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg
        gc.collect()

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat:
            cat_aggregations[cat] = ['mean']
        for cat in bb_cat:
            cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(
            {**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(
            ['BURO_' + e[0] + "_" + e[1].upper()
                for e in bureau_agg.columns.tolist()])
        bureau_agg_pref = bureau.groupby('SK_ID_CURR').head(SUB_HEAD_SIZE).\
            groupby('SK_ID_CURR').\
            agg({**num_aggregations, **cat_aggregations})
        bureau_agg_pref.columns = pd.Index(
            ['BURO_PREF' + e[0] + "_" + e[1].upper()
                for e in bureau_agg_pref.columns.tolist()])
        bureau_agg = bureau_agg.join(bureau_agg_pref, how='left', on='SK_ID_CURR')
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(
            ['ACTIVE_' + e[0] + "_" + e[1].upper()
                for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(
            ['CLOSED_' + e[0] + "_" + e[1].upper()
                for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        return bureau_agg

    # Preprocess POS_CASH_balance.csv
    def fe_pos_cash(self, df):
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
#        for cat in cat_cols:
#            aggregations[cat] = ['mean']

        df_agg = df.groupby('SK_ID_CURR').agg(aggregations)
        df_agg.columns = pd.Index(
            ['POS_' + e[0] + "_" + e[1].upper()
                for e in df_agg.columns.tolist()])
        df_agg_pref = df.groupby('SK_ID_CURR').head(SUB_HEAD_SIZE).\
                groupby('SK_ID_CURR').agg(aggregations)
        df_agg_pref.columns = pd.Index(
            ['POS_PREF_' + e[0] + "_" + e[1].upper()
                for e in df_agg_pref.columns.tolist()])
        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count df cash accounts
        df_agg['POS_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg

    # Preprocess installments_payments.csv
    def fe_installments_payments(self, df):
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        # Percentage and difference paid in each dftallment (amount paid and
        # dftallment value)
        df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
        df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
        df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['mean', 'var'],
            'PAYMENT_DIFF': ['mean', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        df_agg = df.groupby('SK_ID_CURR').agg(aggregations)
        df_agg.columns = pd.Index(
            ['INSTAL_' + e[0] + "_" + e[1].upper()
                for e in df_agg.columns.tolist()])
        df_agg_pref = df.groupby('SK_ID_CURR').head(SUB_HEAD_SIZE).\
                groupby('SK_ID_CURR').agg(aggregations)
        df_agg_pref.columns = pd.Index(
            ['INSTAL_PREF_' + e[0] + "_" + e[1].upper()
                for e in df_agg_pref.columns.tolist()])
        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count dftallments accounts
        df_agg['INSTAL_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg

    # Preprocess credit_card_balance.csv
    def fe_credit_card_balance(self, df):
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        # General aggregations
        df.drop(['SK_ID_PREV'], axis=1, inplace=True)
        df_agg = df.groupby('SK_ID_CURR').agg(
            ['min', 'max', 'mean', 'sum', 'var'])
        df_agg.columns = pd.Index(
            ['CC_' + e[0] + "_" + e[1].upper()
                for e in df_agg.columns.tolist()])
        df_agg_pref = df.groupby('SK_ID_CURR').head(SUB_HEAD_SIZE).\
                groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        df_agg_pref.columns = pd.Index(
            ['CC_PREF_' + e[0] + "_" + e[1].upper()
                for e in df_agg_pref.columns.tolist()])
        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count credit card lines
        df_agg['CC_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg

    def add_impute_required_flg(self, df, null_ratio_thr=0.):
        '''
        add imputing column based on the null ratio.
        default is 0., which means all of the columns which holds NaN
        are added imputation flag.

        '''
        self.logger.info('adding imputing flag to')
        cat_list, dis_num_list, num_list = \
            self.feature_type_split(df)
        self.logger.info('Now imputing {}...'.format(cat_list))
        df = self.impute(
            #            df, cat_list, strategy='most_frequent')
            df, cat_list, strategy='fill')
        self.logger.info('Now imputing {}...'.format(dis_num_list))
        df = self.impute(
            df, dis_num_list, strategy='most_frequent')
        self.logger.info('Now imputing {}...'.format(num_list))
        df = self.impute(
            df, num_list, strategy='median')
        return df

    def impute_train_and_test(self, ):
        self.logger.info('Now imputing ...')
        cat_list, dis_num_list, num_list = \
            self.feature_type_split(self.train_df)
        self.train_df = self.impute(
            self.train_df, cat_list, strategy='fill')
        self.train_df = self.impute(
            self.train_df, dis_num_list, strategy='most_frequent')
        self.train_df = self.impute(
            self.train_df, num_list, strategy='median')
        cat_list, dis_num_list, num_list = \
            self.feature_type_split(self.test_df)
        self.test_df = self.impute(
            self.test_df, cat_list, strategy='fill')
        self.test_df = self.impute(
            self.test_df, dis_num_list, strategy='most_frequent')
        self.test_df = self.impute(
            self.test_df, num_list, strategy='median')

    # feature engineering for previous application
    def add_past_n_raws(self, past_num=5):
        '''
        sort by DAYS_DECISION, and use top past_num info.

        '''
        self.logger.info('adding previous loan nums')
        targets = [
            'NAME_CONTRACT_TYPE',
            'AMT_ANNUITY',
            'AMT_APPLICATION',
            'AMT_CREDIT',
            'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE',
            'WEEKDAY_APPR_PROCESS_START',
            'HOUR_APPR_PROCESS_START',
            'RATE_DOWN_PAYMENT',
            'RATE_INTEREST_PRIMARY',
            'RATE_INTEREST_PRIVILEGED',
            'NAME_CASH_LOAN_PURPOSE',
            'NAME_CONTRACT_STATUS',
            'DAYS_DECISION',
            'NAME_PAYMENT_TYPE',
            'CODE_REJECT_REASON',
            'NAME_TYPE_SUITE',
            'NAME_CLIENT_TYPE',
            'NAME_GOODS_CATEGORY',
            'NAME_PORTFOLIO',
            'NAME_PRODUCT_TYPE',
            'CHANNEL_TYPE',
            'SELLERPLACE_AREA',
            'NAME_SELLER_INDUSTRY',
            'CNT_PAYMENT',
            'NAME_YIELD_GROUP',
            'PRODUCT_COMBINATION',
            'DAYS_FIRST_DRAWING',
            'DAYS_LAST_DUE',
            'DAYS_TERMINATION',
            'NFLAG_INSURED_ON_APPROVAL',
        ]

        special_encodes = {
            'SELLERPLACE_AREA': 'count',
        }

        special_impute = {
            'CNT_PAYMENT': 'zero',
            'HOUR_APPR_PROCESS_START': '-1',
        }

        self.logger.info('targets are {}'.format(targets))

        self.train_df = self.train_df.merge(prev_loan_cnt_df, on='SK_ID_CURR')
        self.train_df.SK_ID_PREV_CNT.fillna(0, inplace=True)
        self.test_df = self.test_df.merge(prev_loan_cnt_df, on='SK_ID_CURR')
        self.test_df.SK_ID_PREV_CNT.fillna(0, inplace=True)

    def add_prev_loan_cnt(self, ):
        self.logger.info('adding previous loan nums')
        prev_loan_cnt_df = self.prev_app_df.groupby(
            'SK_ID_CURR', as_index=False).\
            SK_ID_PREV.count().\
            astype(int).\
            rename(columns={'SK_ID_PREV': 'SK_ID_PREV_CNT'})
        self.train_df = self.train_df.merge(prev_loan_cnt_df, on='SK_ID_CURR')
        self.train_df.SK_ID_PREV_CNT.fillna(0, inplace=True)
        self.test_df = self.test_df.merge(prev_loan_cnt_df, on='SK_ID_CURR')
        self.test_df.SK_ID_PREV_CNT.fillna(0, inplace=True)

    def add_prev_app_diff_features():
        [['AMT_APPLICATION', 'AMT_CREDIT'],
         ]
