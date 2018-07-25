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
        # ===============================
        # 欠損値埋め
        # ===============================
        # application train のみにある categorical features を削除
        #df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        #df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        #df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        #df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

#        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
#        live = [_f for _f in df.columns if ('FLAG_' in _f) &
#                ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
#
        # ===============================
        # group 毎の統計 (target encoding 以外)
        # ===============================
###        inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
###            'ORGANIZATION_TYPE').mean()['AMT_INCOME_TOTAL']
###        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

        # ===============================
        # target encodings (include ext_srcs)
        # ===============================
#        exts = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
#        ext_mean = 'NEW_EXTS_MEAN'
#        df[ext_mean] = df[exts].mean(axis=1)
#        ext_orgs = df[exts + ['ORGANIZATION_TYPE']]\
#                .groupby('ORGANIZATION_TYPE').mean()[exts]
#        for ext in exts:
#            df['NEW_'+ext+'_BY_ORG'] = df['ORGANIZATION_TYPE'].map(ext_orgs[ext])
#        df['NEW_EXT_MEAN_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
#        ext_mean_orgs = df[['ORGANIZATION_TYPE', ext_mean]]\
#                .groupby('ORGANIZATION_TYPE').mean()[ext_mean]
#        df[ext_mean+'_BY_ORG'] = df['ORGANIZATION_TYPE'].map(ext_mean_orgs)
#        ext_mean_orgs = df[['NAME_EDUCATION_TYPE', ext_mean]]\
#                .groupby('NAME_EDUCATION_TYPE').mean()[ext_mean]
#        df[ext_mean+'_BY_EDU'] = df['NAME_EDUCATION_TYPE'].map(ext_mean_orgs)
#        df.drop(ext_mean, axis=1, inplace=True)

        # ===============================
        # manual feature engineering 
        # ===============================
        # この二つの INCOME 系の ratio は cv を下げる。
        # CREDIT, ANNUITY の分布に地域差 (?) があるためと予想
###        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / \
###            df['AMT_INCOME_TOTAL']
###        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / \
###           (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / \
            (1 + df['CNT_CHILDREN'])
        df['NEW_CNT_PARENT_MEMBERS'] = \
            df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
###        df['NEW_GOODS_TO_ANNUITY_RATIO'] = \
###            df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']
        # 何歳で register したか -> 少しだけ improve
        df['NEW_AGE_DAYS_REGISTRATION'] = \
            df['DAYS_BIRTH'] - df['DAYS_REGISTRATION']
        # 就職後どれくらいで車を買ったか (金遣いの粗さ?)
###        df['NEW_CAR_AGE_MINUS_EMPLOYED'] = \
###            df['OWN_CAR_AGE'] - df['DAYS_EMPLOYED']
#        df['NEW_CAR_AGE_DAYS_BIRTH_DIFF'] = \
#            df['OWN_CAR_AGE'] * 365 + df['DAYS_BIRTH']
#        df['NEW_CAR_AGE_EMPLOYED_DIFF'] = \
#            df['OWN_CAR_AGE'] * 365 + df['DAYS_EMPLOYED']
#        df['NEW_CAR_AGE_DAYS_REGISTRATION_DIFF'] = \
#            df['OWN_CAR_AGE'] * 365 + df['DAYS_REGISTRATION']
#        df['NEW_CAR_AGE_DAYS_REGISTRATION_DIFF'] = \
#            df['OWN_CAR_AGE'] * 365 + df['DAYS_REGISTRATION']
#        df['NEW_ID_PUBLISH_REGISTRATION_DIFF'] = \
#            df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']

        # OBS, DEF 系 0.003 程 improve
        df['NEW_DEF/OBS_60'] = \
            df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']
        df['NEW_DEF/OBS_30'] = \
            df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
        df['NEW_60/30_OBS'] = \
            df['OBS_60_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
        df['NEW_60/30_DEF'] = \
            df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['DEF_30_CNT_SOCIAL_CIRCLE']

        # REGION_POPULATION_RELATIVE -> POPRAT で割る系は微妙そう
###        df['NEW_AMT_GOODS_PRICE_POPRAT'] = \
###            df['AMT_GOODS_PRICE'] / df['REGION_POPULATION_RELATIVE']
###        df['NEW_AMT_CREDIT_POPRAT'] = \
###            df['AMT_CREDIT'] * df['REGION_POPULATION_RELATIVE']
###        df['NEW_AMT_ANNUITY_POPRAT'] = \
###            df['AMT_ANNUITY'] * df['REGION_POPULATION_RELATIVE']
###        df['NEW_AMT_GOODS_PRICE_POPRAT'] = \
###            df['AMT_GOODS_PRICE'] * df['REGION_POPULATION_RELATIVE']
#        df['NEW_AMT_INCOME_TOTAL_POPRAT'] = \
#            df['AMT_INCOME_TOTAL'] / df['REGION_RATING_CLIENT']

        # DOCUMENT 数 -> 割と improve
        df['NEW_NUM_DOCS'] = \
           df[df.columns[df.columns.str.contains('FLAG_DOCUMENT_')]].sum(axis=1, skipna=True)
#        df['NEW_SUM_AVGS'] = \
#           df[df.columns[df.columns.str.contains('_AVG$')]].prod(axis=1, skipna=True)
#        df['NEW_SUM_MODES'] = \
#           df[df.columns[df.columns.str.contains('_MODE$')]].sum(axis=1, skipna=True)
#        df['NEW_SUM_MEDIS'] = \
#           df[df.columns[df.columns.str.contains('_MEDI$')]].sum(axis=1, skipna=True)
#        print(df[df.columns[df.columns.str.contains('FLAG_DOCUMENT_')]])
#           df.apply(lambda x: x.str.extract('FLAG_DOCUMENT_(.+)').sum(), axis=1)
           #df[df.str.extract('FLAG_DOCUMENT_(.+)')].sum()

#        df['NEW_PREV_DAYS_TERMINATION_MAX_DAYS_EMPLOYED_DIFF'] =\
#            df['PREV_DAYS_TERMINATION_MAX'] - df['DAYS_EMPLOYED']
#        df['NEW_PREV_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_MAX_DAYS_EMPLOYED_DIFF'] =\
#            df['PREV_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_MAX'] - df['DAYS_EMPLOYED']
#        df['NEW_PREV_DAYS_DECISION_MAX_DAYS_EMPLOYED_DIFF'] =\
#            df['PREV_DAYS_DECISION_MAX'] - df['DAYS_EMPLOYED']
#        df['NEW_BURO_DAYS_CREDIT_MAX_DAYS_EMPLOYED_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] - df['DAYS_EMPLOYED']
#        df['NEW_BURO_DAYS_CREDIT_MAX_DAYS_BIRTH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] - df['DAYS_BIRTH']
#        df['NEW_BURO_DAYS_CREDIT_MAX_DAYS_LAST_PHONE_CHANGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] - df['DAYS_LAST_PHONE_CHANGE']
#        df['NEW_BURO_DAYS_CREDIT_MAX_DAYS_ID_PUBLISH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] - df['DAYS_ID_PUBLISH']
#        df['NEW_BURO_DAYS_CREDIT_MAX_OWN_CAR_AGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] + df['OWN_CAR_AGE'] * 365
#        df['NEW_BURO_DAYS_CREDIT_MAX_DAYS_DAYS_REGISTRATION_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MAX'] - df['DAYS_REGISTRATION']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_DAYS_EMPLOYED_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] - df['DAYS_EMPLOYED']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_DAYS_BIRTH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] - df['DAYS_BIRTH']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_DAYS_LAST_PHONE_CHANGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] - df['DAYS_LAST_PHONE_CHANGE']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_DAYS_ID_PUBLISH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] - df['DAYS_ID_PUBLISH']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_DAYS_DAYS_REGISTRATION_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] - df['DAYS_REGISTRATION']
#        df['NEW_BURO_DAYS_CREDIT_MEAN_OWN_CAR_AGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MEAN'] + df['OWN_CAR_AGE'] * 365
#        df['NEW_BURO_DAYS_CREDIT_MIN_DAYS_EMPLOYED_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] - df['DAYS_EMPLOYED']
#        df['NEW_BURO_DAYS_CREDIT_MIN_DAYS_BIRTH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] - df['DAYS_BIRTH']
#        df['NEW_BURO_DAYS_CREDIT_MIN_DAYS_LAST_PHONE_CHANGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] - df['DAYS_LAST_PHONE_CHANGE']
#        df['NEW_BURO_DAYS_CREDIT_MIN_DAYS_ID_PUBLISH_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] - df['DAYS_ID_PUBLISH']
#        df['NEW_BURO_DAYS_CREDIT_MIN_DAYS_DAYS_REGISTRATION_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] - df['DAYS_REGISTRATION']
#        df['NEW_BURO_DAYS_CREDIT_MIN_OWN_CAR_AGE_DIFF'] = \
#            df['BURO_DAYS_CREDIT_MIN'] + df['OWN_CAR_AGE'] * 365

        # 人口密度に対する家に関する統計量
        house_stat_list = [
#                'APARTMENTS_AVG',
#                'BASEMENTAREA_AVG',
                'YEARS_BEGINEXPLUATATION_AVG',
#                'YEARS_BUILD_AVG',
#                'COMMONAREA_AVG',
#                'ELEVATORS_AVG',
#                'ENTRANCES_AVG',
#                'FLOORSMAX_AVG',
#                'FLOORSMIN_AVG',
#                'LANDAREA_AVG',
#                'LIVINGAPARTMENTS_AVG',
#                'LIVINGAREA_AVG',
#                'NONLIVINGAPARTMENTS_AVG',
#                'NONLIVINGAREA_AVG',
                'APARTMENTS_MODE', #
                'BASEMENTAREA_MODE', #
                'YEARS_BEGINEXPLUATATION_MODE',
                'YEARS_BUILD_MODE', #
                'COMMONAREA_MODE',
                'ELEVATORS_MODE', #
                'ENTRANCES_MODE',
                'FLOORSMAX_MODE',
                'FLOORSMIN_MODE',
                'LANDAREA_MODE',
                'LIVINGAPARTMENTS_MODE',
                'LIVINGAREA_MODE',
                'NONLIVINGAPARTMENTS_MODE',
                'NONLIVINGAREA_MODE', #
#                'APARTMENTS_MEDI',
                'BASEMENTAREA_MEDI',
                'YEARS_BEGINEXPLUATATION_MEDI',
#                'YEARS_BUILD_MEDI',
#                'COMMONAREA_MEDI',
#                'ELEVATORS_MEDI',
#                'ENTRANCES_MEDI',
#                'FLOORSMAX_MEDI',
#                'FLOORSMIN_MEDI',
                'LANDAREA_MEDI',
#                'LIVINGAPARTMENTS_MEDI',
#                'LIVINGAREA_MEDI',
#                'NONLIVINGAPARTMENTS_MEDI',
#                'NONLIVINGAREA_MEDI',
#                'TOTALAREA_MODE',
        ]

#        for house_stat in house_stat_list:
#            df['NEW_'+house_stat+'_REGRATRAT'] = \
#                    df[house_stat] / df['AMT_INCOME_TOTAL']

##        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
##        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
#        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
#        df['NEW_EMPLOY_TO_BIRTH_RATIO'] = \
#            df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
#        # importnant
#        df['NEW_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * \
#            df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
#        # important
###        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1',
###                                         'EXT_SOURCE_2',
###                                         'EXT_SOURCE_3']].mean(axis=1)
###        df['NEW_EXT_SCORES_STD'] = df[['EXT_SOURCE_1',
###                                       'EXT_SOURCE_2',
###                                       'EXT_SOURCE_3']].std(axis=1)
##        df['NEW_EXT_SOURCE_1M2'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
##        df['NEW_EXT_SOURCE_2M3'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
##        df['NEW_EXT_SOURCE_3M1'] = df['EXT_SOURCE_3'] - df['EXT_SOURCE_1']
#        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
#        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
#        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / \
#            df['DAYS_BIRTH']
#        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = \
#            df['DAYS_LAST_PHONE_CHANGE'] / \
#            df['DAYS_EMPLOYED']

        # Categorical features with Binary encode (0 or 1; two categories)
#        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
#            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # Categorical features with One-Hot encode
        dropcolumn = [
                     'FLAG_DOCUMENT_2', 
#                     'FLAG_DOCUMENT_4',
#                     'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
#                     'FLAG_DOCUMENT_7',
#                     'FLAG_DOCUMENT_8',
#                     'FLAG_DOCUMENT_9', 
                     'FLAG_DOCUMENT_10',
#                     'FLAG_DOCUMENT_11', 
                     'FLAG_DOCUMENT_12',
                     'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                     'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                     'FLAG_DOCUMENT_17',
#                     'FLAG_DOCUMENT_18',
                     'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                     'FLAG_DOCUMENT_21']
        dropcolumn += [
#                'NAME_CONTRACT_TYPE', # これが overfit の原因 ?
#                'AMT_CREDIT',
#                'AMT_REQ_CREDIT_BUREAU_DAY',
###                'DAYS_REGISTRATION',
###                'REGION_POPULATION_RELATIVE',
###                'AMT_REQ_CREDIT_BUREAU_YEAR',
###                'HOUR_APPR_PROCESS_START',
###                'TOTALAREA_MODE',
#                'NONLIVINGAREA_AVG',
###                'NAME_EDUCATION_TYPE',
###                'BASEMENTAREA_AVG',
###                'FLAG_WORK_PHONE',
###                'YEARS_BUILD_MODE',
###                'ORGANIZATION_TYPE',
###                'NAME_INCOME_TYPE',
###                'FLAG_MOBIL',
#                'FLAG_CONT_MOBILE',
#                'HOUSETYPE_MODE',
#                'WALLSMATERIAL_MODE',
#                'FLAG_EMP_PHONE',
#                'FLAG_DOCUMENT_9',
#                'OCCUPATION_TYPE',
#                'AMT_REQ_CREDIT_BUREAU_HOUR',
#                'FLAG_OWN_CAR',
#                'REG_REGION_NOT_WORK_REGION',
#                'FLAG_EMAIL',
#                'CNT_CHILDREN',
#                'CNT_FAM_MEMBERS',
#                'WEEKDAY_APPR_PROCESS_START',
#                'DEF_60_CNT_SOCIAL_CIRCLE',
#                'DEF_30_CNT_SOCIAL_CIRCLE',
#                'OBS_30_CNT_SOCIAL_CIRCLE',
#                'OBS_60_CNT_SOCIAL_CIRCLE',
                ]
        dropcolumn += house_stat_list
        df = df.drop(dropcolumn, axis=1)
        gc.collect()
        return df

    def fe_application_prev(self, df):
        df_for_cat_tail = df[['CC_PREV_NAME_CONTRACT_STATUS_TAIL']]
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
#        df, cat_cols = self.onehot_encoding(df, drop_first=False,
#                special_list=[
#                    'NAME_CONTRACT_TYPE',
#                    'FLAG_LAST_APPL_IN_DAY',
#                    'NAME_CONTRACT_STATUS',
#                    'NAME_PAYMENT_TYPE',
#                    'NAME_PORTFOLIO',
#                    'NAME_YIELD_GROUP',
#                    'PRODUCT_COMBINATION',
#                    ])

        # ===============================
        # 欠損値埋め
        # ===============================
        # Days 365.243 values -> nan
        df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

        # ===============================
        # manual feature engineering
        # ===============================
        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = \
            df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
        df['NEW_RATE_INTEREST_RATE'] = \
            df['RATE_INTEREST_PRIMARY'] / df['RATE_INTEREST_PRIVILEGED']
        df['NEW_DAYS_FIRST_DUE_DIFF'] = \
            df['DAYS_FIRST_DUE'] - df['DAYS_FIRST_DRAWING']
        df['NEW_DAYS_LAST_DUE_DIFF'] = \
            df['DAYS_LAST_DUE'] - df['DAYS_LAST_DUE_1ST_VERSION']
        df['NEW_DAYS_FIRST_AND_LAST_DUE_DIFF'] = \
            df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_FIRST_DUE']
        df['NEW_CREDIT_SELLERPLACE_RATE'] = \
            df['AMT_CREDIT'] / df['SELLERPLACE_AREA']

        # prev features from pos
        df['NEW_POS_PREV_INSTALMENT_SPEED'] = \
            df['POS_PREV_CNT_INSTALMENT_FUTURE_SIZE'] /\
            df['POS_PREV_CNT_INSTALMENT_FUTURE_MAX']

        # prev features from instal
        tail_0_mask = df.INSTAL_PREV_NUM_INSTALMENT_VERSION_TAIL == 0
        tail_1_mask = df.INSTAL_PREV_NUM_INSTALMENT_VERSION_TAIL == 1
        tail_2_mask = df.INSTAL_PREV_NUM_INSTALMENT_VERSION_TAIL == 2
        tail_others_mask = df.INSTAL_PREV_NUM_INSTALMENT_VERSION_TAIL > 2
        df['NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_0'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_0_mask
        df['NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_1'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_1_mask
        df['NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_2'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_2_mask
        df['NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_OTHERS'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_others_mask
        df['NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_0'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_0_mask
        df['NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_1'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_1_mask
        df['NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_2'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_2_mask
        df['NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_OTHERS'] = \
            df.INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX * tail_others_mask
        df['NEW_INSTAL_PREV_AMT_INSTALMENT_AND_PAYMENT_SUM_DIFF'] = \
            df['INSTAL_PREV_AMT_INSTALMENT_SUM'] - df['INSTAL_PREV_AMT_PAYMENT_SUM']
        df['NEW_INSTAL_PREV_AMT_INSTALMENT_AND_PAYMENT_SUM_RATIO'] = \
            df['INSTAL_PREV_AMT_PAYMENT_SUM'] / df['INSTAL_PREV_AMT_INSTALMENT_SUM']

        # prev features from cc
        tail_completed_mask = df_for_cat_tail.CC_PREV_NAME_CONTRACT_STATUS_TAIL == 'Completed'
        tail_active_mask = df_for_cat_tail.CC_PREV_NAME_CONTRACT_STATUS_TAIL == 'Active'
        df['NEW_CC_PREV_AMT_BALANCE_MAX_MIN_DIFF'] = \
            df['CC_PREV_AMT_BALANCE_MAX'] - df['CC_PREV_AMT_BALANCE_MIN']
        df['NEW_CC_PREV_AMT_BALANCE_MAX_HEAD_DIFF'] = \
            df['CC_PREV_AMT_BALANCE_MAX'] - df['CC_PREV_AMT_BALANCE_HEAD']
        df['NEW_CC_PREV_AMT_BALANCE_MAX_TAIL_DIFF'] = \
            df['CC_PREV_AMT_BALANCE_MAX'] - df['CC_PREV_AMT_BALANCE_HEAD']
        df['NEW_CC_PREV_AMT_BALANCE_MIN_HEAD_DIFF'] = \
            df['CC_PREV_AMT_BALANCE_MIN'] - df['CC_PREV_AMT_BALANCE_TAIL']
        df['NEW_CC_PREV_AMT_BALANCE_MIN_TAIL_DIFF'] = \
            df['CC_PREV_AMT_BALANCE_MIN'] - df['CC_PREV_AMT_BALANCE_TAIL']
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_MIN_DIFF'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX'] - df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN']
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_HEAD_DIFF'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX'] - df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_HEAD']
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_TAIL_DIFF'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX'] - df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_HEAD']
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN_HEAD_DIFF'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN'] - df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_TAIL']
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN_TAIL_DIFF'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN'] - df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_TAIL']
        df['NEW_CC_PREV_SIZE_AND_NUNIQUE_RATIO'] = \
            df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_NUNIQUE'] / df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_SIZE']
        df['NEW_CC_PREV_AMT_DRAWINGS_SUM_RATIO'] = \
            df['CC_PREV_AMT_DRAWINGS_ATM_CURRENT_SUM'] / df['CC_PREV_AMT_DRAWINGS_CURRENT_SUM']
        df['NEW_CC_PREV_AMT_INST_MIN_REGULARITY_MAX_MIN_DIFF'] =\
            df['CC_PREV_AMT_INST_MIN_REGULARITY_MAX'] - df['CC_PREV_AMT_INST_MIN_REGULARITY_MIN']
        df['NEW_CC_PREV_AMT_PAYMENT_CURRENT_AND_TOTAL_DIFF'] =\
            df['CC_PREV_AMT_PAYMENT_CURRENT_MEAN'] - df['CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MEAN']
        df['NEW_CC_PREV_AMT_BALANCE_PAYMENT_TOTAL_CURRENT_DIFF'] =\
            df['CC_PREV_AMT_BALANCE_MEAN'] - df['CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MEAN']
        df['NEW_CC_PREV_AMT_BALANCE_PAYMENT_TOTAL_CURRENT_RATIO'] =\
            df['CC_PREV_AMT_BALANCE_MEAN'] / (df['CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MEAN'] + 1)
        df['NEW_CC_PREV_COMPLETED_TAIL_MONTHS_BALANCE'] =\
            df.CC_PREV_MONTHS_BALANCE_TAIL * tail_completed_mask
        df['NEW_CC_PREV_ACTIVE_TAIL_MONTHS_BALANCE'] =\
            df.CC_PREV_MONTHS_BALANCE_TAIL * tail_active_mask
        df['NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_SIZE_INS_MATURE_CUM_TAIL_RATIO'] = \
            df['CC_PREV_CNT_INSTALMENT_MATURE_CUM_TAIL'] / df['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_SIZE']

        # Add feature: value ask / value received percentage
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'NFLAG_LAST_APPL_IN_DAY': ['mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'RATE_INTEREST_PRIMARY': ['max', 'mean'],
            'RATE_INTEREST_PRIVILEGED': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean', 'min'],
            'DAYS_TERMINATION': ['max', 'mean', 'min'],
            'SELLERPLACE_AREA': ['max', 'mean', 'min'],
            'CNT_PAYMENT': ['mean'],
            'NFLAG_INSURED_ON_APPROVAL': ['mean'],
            'NEW_CREDIT_TO_ANNUITY_RATIO': ['max', 'mean'],
            'NEW_CREDIT_TO_GOODS_RATIO': ['max', 'mean'],
            'NEW_APP_CREDIT_PERC': ['max', 'mean'],
            'NEW_RATE_INTEREST_RATE': ['max', 'mean'],
            'NEW_DAYS_FIRST_DUE_DIFF': ['min', 'mean', 'max'],
            'NEW_DAYS_LAST_DUE_DIFF': ['min', 'mean', 'max'],
            'NEW_DAYS_FIRST_AND_LAST_DUE_DIFF': ['min', 'mean', 'max'],
            'NEW_CREDIT_SELLERPLACE_RATE': ['min', 'mean', 'max'],

            'POS_PREV_MONTHS_BALANCE_MAX': ['max', 'mean'],
            'POS_PREV_MONTHS_BALANCE_SIZE': ['max', 'mean'],
            'POS_PREV_MONTHS_BALANCE_MIN': ['min', 'max'],
            'POS_PREV_MONTHS_BALANCE_<LAMBDA>': ['max', 'mean'],
            'POS_PREV_SK_DPD_MAX': ['max', 'mean'],
            'POS_PREV_SK_DPD_MEAN': ['max', 'mean', 'min'],
            'POS_PREV_SK_DPD_DEF_MAX': ['max', 'mean'],
            'POS_PREV_SK_DPD_DEF_MEAN': ['max', 'mean', 'min'],
            'POS_PREV_CNT_INSTALMENT_NUNIQUE': ['max', 'mean', 'min'],
            'POS_PREV_CNT_INSTALMENT_FUTURE_SIZE': ['mean'],
            'POS_PREV_CNT_INSTALMENT_FUTURE_MAX': ['max'],
            'POS_PREV_NEW_SK_DPD_DIFF_MAX': ['max', 'mean'],
            'POS_PREV_NEW_SK_DPD_DIFF_MEAN': ['mean'],
            'POS_PREV_NEW_SK_DPD_DIFF_SUM': ['mean', 'sum'],
            'NEW_POS_PREV_INSTALMENT_SPEED': ['mean', 'max', 'min'],

            'INSTAL_PREV_NUM_INSTALMENT_VERSION_MEDIAN': ['median'],
            'INSTAL_PREV_AMT_INSTALMENT_MEAN': ['max', 'mean', ],
            'INSTAL_PREV_AMT_PAYMENT_MEAN': ['max', 'mean', ],
            'INSTAL_PREV_DAYS_INSTALMENT_MAX': ['max', 'mean', 'min'],
            'INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX': ['max', 'mean', 'min'],
            'INSTAL_PREV_NEW_PAYMENT_PERC_MEAN': ['mean', 'min'],
            'INSTAL_PREV_NEW_PAYMENT_PERC_MIN': ['max', 'mean', 'min'],
            'INSTAL_PREV_NEW_PAYMENT_PERC_VAR': ['mean'],
            'INSTAL_PREV_NEW_PAYMENT_DIFF_MEAN': ['mean', 'min'],
            'INSTAL_PREV_NEW_PAYMENT_DIFF_VAR': ['mean'],
            'INSTAL_PREV_NEW_DAYS_PAYMENT_DIFF_MAX': ['max', 'mean'],
            'INSTAL_PREV_NEW_DAYS_PAYMENT_DIFF_MEAN': ['mean'],
            'INSTAL_PREV_NEW_DAYS_PAYMENT_DIFF_MIN': ['min', 'max'],
            'NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_0': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_1': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_2': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_OTHERS': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_0': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_1': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_2': ['mean', 'min'],
            'NEW_INSTAL_PREV_DAYS_INSTALMENT_MAX_FOR_NUM_INSTALMENT_VERSION_TAIL_OTHERS': ['mean', 'min'],
            'NEW_INSTAL_PREV_AMT_INSTALMENT_AND_PAYMENT_SUM_DIFF': ['max', 'mean', 'sum'],
            'NEW_INSTAL_PREV_AMT_INSTALMENT_AND_PAYMENT_SUM_RATIO': ['mean', 'min'],
            'CC_PREV_MONTHS_BALANCE_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_BALANCE_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_BALANCE_MEAN': ['mean'],
            'CC_PREV_AMT_BALANCE_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_NUNIQUE': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_ATM_CURRENT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_ATM_CURRENT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_ATM_CURRENT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_ATM_CURRENT_VAR': ['max', 'mean', ],
            'CC_PREV_AMT_DRAWINGS_ATM_CURRENT_SUM': ['mean', 'sum'],
            'CC_PREV_AMT_DRAWINGS_CURRENT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_CURRENT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_CURRENT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_CURRENT_VAR': ['max', 'mean', ],
            'CC_PREV_AMT_DRAWINGS_CURRENT_SUM': ['mean', 'sum', ],
            'CC_PREV_AMT_DRAWINGS_OTHER_CURRENT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_OTHER_CURRENT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_OTHER_CURRENT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_OTHER_CURRENT_VAR': ['max', 'mean', ],
            'CC_PREV_AMT_DRAWINGS_OTHER_CURRENT_SUM': ['mean', 'sum', ],
            'CC_PREV_AMT_DRAWINGS_POS_CURRENT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_POS_CURRENT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_POS_CURRENT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_DRAWINGS_POS_CURRENT_VAR': ['max', 'mean', ],
            'CC_PREV_AMT_DRAWINGS_POS_CURRENT_SUM': ['mean', 'sum', ],
            'CC_PREV_AMT_INST_MIN_REGULARITY_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_INST_MIN_REGULARITY_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_INST_MIN_REGULARITY_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_PAYMENT_TOTAL_CURRENT_VAR': ['max', 'mean', 'min'],
            'CC_PREV_AMT_RECEIVABLE_PRINCIPAL_MIN': ['max', 'mean', 'min'],
            'CC_PREV_AMT_RECEIVABLE_PRINCIPAL_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_DIFF_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_DIFF_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_DIFF_MIN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_ATM_CURRENT_PER_CNT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_ATM_CURRENT_PER_CNT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_ATM_CURRENT_PER_CNT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_CURRENT_PER_CNT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_CURRENT_PER_CNT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_CURRENT_PER_CNT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_POS_CURRENT_PER_CNT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_POS_CURRENT_PER_CNT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_POS_CURRENT_PER_CNT_MIN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_OTHER_CURRENT_PER_CNT_MAX': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_OTHER_CURRENT_PER_CNT_MEAN': ['max', 'mean', 'min'],
            'CC_PREV_NEW_AMT_DRAWINGS_OTHER_CURRENT_PER_CNT_MIN': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_MAX_MIN_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_MAX_HEAD_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_MAX_TAIL_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_MIN_HEAD_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_MIN_TAIL_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_MIN_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_HEAD_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MAX_TAIL_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN_HEAD_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_MIN_TAIL_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_SIZE_AND_NUNIQUE_RATIO': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_DRAWINGS_SUM_RATIO': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_INST_MIN_REGULARITY_MAX_MIN_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_PAYMENT_CURRENT_AND_TOTAL_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_PAYMENT_TOTAL_CURRENT_DIFF': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_BALANCE_PAYMENT_TOTAL_CURRENT_RATIO': ['max', 'mean', 'min'],
            'NEW_CC_PREV_COMPLETED_TAIL_MONTHS_BALANCE': ['max', 'mean', 'min'],
            'NEW_CC_PREV_ACTIVE_TAIL_MONTHS_BALANCE': ['max', 'mean', 'min'],
            'NEW_CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_SIZE_INS_MATURE_CUM_TAIL_RATIO': ['max', 'mean', 'min'],
        }

        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']
        self.logger.info('aggregating for SK_ID_CURR...')
        df_agg = df.groupby('SK_ID_CURR')\
            .agg({**num_aggregations, **cat_aggregations})
        df_agg.columns = pd.Index(
            ['PREV_' + e[0] + "_" + e[1].upper()
             for e in df_agg.columns.tolist()])

        # previous Applications: Approved Applications - only numerical features
        approved = df[df['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR')\
            .agg(num_aggregations)
#        app_agg_cols = approved_agg.columns.tolist()
        approved_agg.columns = pd.Index(
            ['APPROVED_' + e[0] + "_" + e[1].upper()
             for e in approved_agg.columns.tolist()])
#        df_agg = approved_agg
        df_agg = df_agg.merge(approved_agg, how='left', on='SK_ID_CURR')

        # previous Applications: Refused Applications - only numerical features
#        refused = df[df['NAME_CONTRACT_STATUS_Refused'] == 1]
#        refused_agg = refused.groupby('SK_ID_CURR')\
#            .agg(num_aggregations)
#        refused_agg.columns = pd.Index(
#            ['REFUSED_' + e[0] + "_" + e[1].upper()
#             for e in refused_agg.columns.tolist()])
#        df_agg = df_agg.join(refused_agg, how='left', on='SK_ID_CURR')

        # ===============================
        # features based on agg or w/o agg
        # ===============================
        df_agg['PREV_NEW_CNT'] = df.groupby('SK_ID_CURR').size()
        df_agg['PREV_NEW_APPROVED_CNT'] = approved.groupby('SK_ID_CURR').size()
#        df_agg['PREV_NEW_REFUSED_CNT'] = refused.groupby('SK_ID_CURR').size()
        df_agg['PREV_NEW_APPROVED_RATIO'] = \
            df_agg['PREV_NEW_APPROVED_CNT'] / df_agg['PREV_NEW_CNT']
#        del refused, refused_agg, approved, approved_agg, df
        del approved, approved_agg, df

        gc.collect()
        return df_agg

    # Preprocess POS_CASH_balance.csv
    def fe_pos_cash(self, df):
        '''
        months balance の tail で status 等の agg

        '''
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        df = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])

        # ===============================
        # manual feature engineering
        # ===============================
        df['NEW_SK_DPD_DIFF'] = \
            df['SK_DPD'] - df['SK_DPD_DEF']

        # Features
        aggregations_curr = {
            'MONTHS_BALANCE': ['mean', ],
            'SK_DPD': ['mean'],
            'SK_DPD_DEF': ['mean'],
            'NEW_SK_DPD_DIFF': ['max', 'mean', 'sum']
        }

        aggregations_prev = {
            'MONTHS_BALANCE': ['max', 'size', 'min', lambda x: x.diff().max()],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean'],
            'CNT_INSTALMENT': ['nunique'],
            'CNT_INSTALMENT_FUTURE': ['size', 'max'],
            'NEW_SK_DPD_DIFF': ['max', 'mean', 'sum']
        }

        for cat in cat_cols:
            aggregations_curr[cat] = ['mean']
            aggregations_prev[cat] = ['mean']

        self.logger.info('aggregating for SK_ID_CURR...')
        df_agg_curr = df.groupby('SK_ID_CURR').agg(aggregations_curr)
        df_agg_curr.columns = pd.Index(
            ['POS_' + e[0] + "_" + e[1].upper()
                for e in df_agg_curr.columns.tolist()])
        # Count df cash accounts
        #df_agg_curr['POS_NEW_COUNT'] = df.groupby('SK_ID_CURR').size()

        self.logger.info('aggregating for SK_ID_PREV_...')
        df_agg_prev = df.groupby('SK_ID_PREV').agg(aggregations_prev)
        df_agg_prev.columns = pd.Index(
            ['POS_PREV_' + e[0] + "_" + e[1].upper()
                for e in df_agg_prev.columns.tolist()])
        # Count df cash accounts
        #df_agg_prev['POS_PREV_NEW_COUNT'] = df.groupby('SK_ID_PREV').size()

        del df
        gc.collect()
        return df_agg_curr, df_agg_prev

    # Preprocess installments_payments.csv
    def fe_installments_payments(self, df):
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        df = df.sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])

        # ===============================
        # manual feature engineering
        # ===============================
        # Percentage and difference paid in each dftallment (amount paid and
        # dftallment value)
        # 最後の version とその時の金額は大事っぽい
        # version 2, 3 は一括返済?
        df['NEW_PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['NEW_PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
        df['NEW_DAYS_PAYMENT_DIFF'] = \
            df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']

        # Features: Perform aggregations
        aggregations_curr = {
        }

        aggregations_prev = {
            'NUM_INSTALMENT_VERSION': ['median'],
            #'NUM_INSTALMENT_VERSION': ['median', lambda x: x.tail(1)],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['max', 'mean', 'sum'],
            'DAYS_INSTALMENT': ['max', 'min'],
            'DAYS_ENTRY_PAYMENT': ['max', 'min'],
            'NEW_PAYMENT_PERC': ['mean', 'min', 'var'],
            'NEW_PAYMENT_DIFF': ['mean', 'var'],
            'NEW_DAYS_PAYMENT_DIFF': ['max', 'mean', 'min'],
        }

        for cat in cat_cols:
            aggregations_curr[cat] = ['mean']
            aggregations_prev[cat] = ['mean']
        df_agg_curr = df.groupby('SK_ID_CURR').agg(aggregations_curr)
        df_agg_curr.columns = pd.Index(
            ['INSTAL_' + e[0] + "_" + e[1].upper()
                for e in df_agg_curr.columns.tolist()])
        df_agg_prev = df.groupby('SK_ID_PREV').agg(aggregations_prev)
        df_agg_prev.columns = pd.Index(
            ['INSTAL_PREV_' + e[0] + "_" + e[1].upper()
                for e in df_agg_prev.columns.tolist()])
        df_agg_prev['INSTAL_PREV_NUM_INSTALMENT_VERSION_TAIL'] = df.groupby('SK_ID_PREV').NUM_INSTALMENT_VERSION.tail(1).astype('int')
#        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count dftallments accounts
#        df_agg_curr['INSTAL_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg_curr, df_agg_prev

    # Preprocess credit_card_balance.csv
    def fe_credit_card_balance(self, df):
        '''
        謎が多いので後で書き直す

        '''
        df_for_cat_tail = df[['SK_ID_PREV', 'NAME_CONTRACT_STATUS']]
        df, cat_cols = self.onehot_encoding(df, drop_first=False)
        df = df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
        # ===============================
        # manual feature engineering
        # ===============================
        df['NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_DIFF'] = df['AMT_BALANCE'] - df['AMT_CREDIT_LIMIT_ACTUAL']
        df['NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_RATIO'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        df['NEW_AMT_RECEIVABLE_DIFF_W_PRINCIPAL'] = df['AMT_RECIVABLE'] - df['AMT_RECEIVABLE_PRINCIPAL']
        df['NEW_AMT_RECEIVABLE_RATIO_W_PRINCIPAL'] = df['AMT_RECEIVABLE_PRINCIPAL'] / (df['AMT_RECIVABLE'] + 1)
        df['NEW_AMT_RECEIVABLE_DIFF_W_TOTAL'] = df['AMT_RECIVABLE'] - df['AMT_TOTAL_RECEIVABLE']
        df['NEW_AMT_RECEIVABLE_RATIO_W_TOTAL'] = df['AMT_RECIVABLE'] / (df['AMT_TOTAL_RECEIVABLE'] + 1)
        df['NEW_AMT_DRAWINGS_DIFF_W_PAYMENT_CURRENT'] = df['AMT_DRAWINGS_CURRENT'] - df['AMT_PAYMENT_CURRENT']
        df['NEW_AMT_DRAWINGS_RATIO_W_PAYMENT_CURRENT'] = df['AMT_DRAWINGS_CURRENT'] / (df['AMT_PAYMENT_CURRENT'] + 1)
        df['NEW_AMT_DRAWINGS_ATM_CURRENT_PER_CNT'] = df['AMT_DRAWINGS_ATM_CURRENT'] / (df['CNT_DRAWINGS_ATM_CURRENT'] + 1)
        df['NEW_AMT_DRAWINGS_CURRENT_PER_CNT'] = df['AMT_DRAWINGS_CURRENT'] / (df['CNT_DRAWINGS_CURRENT'] + 1)
        df['NEW_AMT_DRAWINGS_POS_CURRENT_PER_CNT'] = df['AMT_DRAWINGS_POS_CURRENT'] / (df['CNT_DRAWINGS_POS_CURRENT'] + 1)
        df['NEW_AMT_DRAWINGS_OTHER_CURRENT_PER_CNT'] = df['AMT_DRAWINGS_OTHER_CURRENT'] / (df['CNT_DRAWINGS_OTHER_CURRENT'] + 1)
        df['NEW_SK_DPD_DIFF'] = df['SK_DPD'] - df['SK_DPD_DEF']

        aggregations_curr = {
            'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'var'],
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'var'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'var'],
            'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'var'],
            'SK_DPD': ['max', 'mean', 'var'],
            'SK_DPD_DEF': ['max', 'mean', 'var'],
            'NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_RATIO': ['min', 'mean', 'max'],
            'NEW_AMT_RECEIVABLE_DIFF_W_PRINCIPAL': ['max', 'mean', 'min'],
            'NEW_AMT_RECEIVABLE_RATIO_W_PRINCIPAL': ['max', 'mean', 'min'],
            'NEW_AMT_RECEIVABLE_DIFF_W_TOTAL': ['max', 'mean', 'min'],
            'NEW_AMT_RECEIVABLE_RATIO_W_TOTAL': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_DIFF_W_PAYMENT_CURRENT': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_RATIO_W_PAYMENT_CURRENT': ['max', 'mean', 'min'],
            'NEW_SK_DPD_DIFF': ['max', 'mean', 'var'],
        }

        aggregations_prev = {
            'MONTHS_BALANCE': ['min'],
            'AMT_BALANCE': ['max', 'mean', 'min'], # max - min, max or min - head and tail
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'min', 'nunique', 'size'], # max - min, max or min - head and tail, size / nunique, balance / limit
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'min', 'var', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'min', 'var', 'sum'], # この sum と atm sum の ratio
            'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'min', 'var', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'min', 'var', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['min', 'mean', 'max'], # これらの間の差分等
            'AMT_PAYMENT_CURRENT': ['max', 'mean', 'min', 'var'], # これと totol の差分等
            'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'min', 'var'], # これと balance の ratio
            'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max'],
            'NEW_AMT_BALANCE_CREDIT_LIMIT_ACTUAL_DIFF': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_ATM_CURRENT_PER_CNT': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_CURRENT_PER_CNT': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_POS_CURRENT_PER_CNT': ['max', 'mean', 'min'],
            'NEW_AMT_DRAWINGS_OTHER_CURRENT_PER_CNT': ['max', 'mean', 'min'],
        }

        for cat in cat_cols:
            aggregations_curr[cat] = ['mean']
            aggregations_prev[cat] = ['mean']

        df_agg_curr = df.groupby('SK_ID_CURR').agg(aggregations_curr)
        df_agg_curr.columns = pd.Index(
            ['CC' + e[0] + "_" + e[1].upper()
                for e in df_agg_curr.columns.tolist()])
        df_agg_prev = df.groupby('SK_ID_PREV').agg(aggregations_prev)
        df_agg_prev.columns = pd.Index(
            ['CC_PREV_' + e[0] + "_" + e[1].upper()
                for e in df_agg_prev.columns.tolist()])

        # Count credit card lines
        #df_agg['CC_COUNT'] = df.groupby('SK_ID_CURR').size()
        df_agg_prev['CC_PREV_MONTHS_BALANCE_TAIL'] = df.groupby('SK_ID_PREV')\
            .MONTHS_BALANCE.tail(1).astype('int')
        df_agg_prev['CC_PREV_AMT_BALANCE_HEAD'] = df.groupby('SK_ID_PREV')\
            .MONTHS_BALANCE.head(1)
        df_agg_prev['CC_PREV_AMT_BALANCE_TAIL'] = df.groupby('SK_ID_PREV')\
            .MONTHS_BALANCE.tail(1)
        df_agg_prev['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_HEAD'] = df.groupby('SK_ID_PREV')\
            .AMT_CREDIT_LIMIT_ACTUAL.head(1)
        df_agg_prev['CC_PREV_AMT_CREDIT_LIMIT_ACTUAL_TAIL'] = df.groupby('SK_ID_PREV')\
            .AMT_CREDIT_LIMIT_ACTUAL.tail(1)
        df_agg_prev['CC_PREV_CNT_INSTALMENT_MATURE_CUM_TAIL'] = df.groupby('SK_ID_PREV')\
            .CNT_INSTALMENT_MATURE_CUM.tail(1)
        df_agg_prev['CC_PREV_NAME_CONTRACT_STATUS_TAIL'] = df_for_cat_tail.groupby('SK_ID_PREV')\
            .NAME_CONTRACT_STATUS.tail(1)

        del df
        gc.collect()
        return df_agg_curr, df_agg_prev

    # Preprocess bureau.csv and bureau_balance.csv
    def fe_bureau_and_balance(self, bureau, bb):
        bb = bb.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])
        bb_for_cat_tail = bb[['SK_ID_BUREAU', 'STATUS']]
        bb, bb_cat = self.onehot_encoding(bb, drop_first=False)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                                   for e in bb_agg.columns.tolist()])
        bb_agg['BB_STATUS_HEAD'] = bb_for_cat_tail.groupby(['SK_ID_BUREAU']).head(1).STATUS
        bb_agg['BB_STATUS_TAIL'] = bb_for_cat_tail.groupby(['SK_ID_BUREAU']).tail(1).STATUS

        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau, bureau_cat = self.onehot_encoding(bureau, drop_first=False)
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg
        gc.collect()

        # ===============================
        # manual feature engineering
        # ===============================
        bureau['NEW_BURO_DAYS_CREDIT_DAYS_CREDIT_ENDDATE_DIFF'] = \
            bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
        bureau['NEW_BURO_DAYS_CREDIT_DAYS_ENDDATE_FACT_DIFF'] = \
            bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
        bureau['NEW_BURO_DAYS_CREDIT_DAYS_CREDIT_ENDDATE_DIFF'] = \
            bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
        bureau['NEW_BURO_AMT_CREDIT_MAX_OVERDUE_CREDIT_DAY_OVERDUE_PROD'] = \
            bureau['AMT_CREDIT_MAX_OVERDUE'] * bureau['CREDIT_DAY_OVERDUE']
        bureau['NEW_BURO_CNT_CREDIT_PROLONG_CREDIT_DAY_OVERDUE_PROD'] = \
            bureau['CNT_CREDIT_PROLONG'] * bureau['CREDIT_DAY_OVERDUE']
        bureau['NEW_BURO_AMT_CREDIT_MAX_OVERDUE_CNT_CREDIT_PROLONG_PROD'] = \
            bureau['AMT_CREDIT_MAX_OVERDUE'] * bureau['CNT_CREDIT_PROLONG']
        bureau['NEW_BURO_AMT_CREDIT_SUM_CREDIT_DAY_OVERDUE_PROD'] = \
            bureau['AMT_CREDIT_SUM'] * bureau['CREDIT_DAY_OVERDUE']
        bureau['NEW_BURO_AMT_CREDIT_SUM_CNT_CREDIT_PROLONG_PROD'] = \
            bureau['AMT_CREDIT_SUM'] * bureau['CNT_CREDIT_PROLONG']
        bureau['NEW_BURO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM_RATIO'] = \
            bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
        bureau['NEW_BURO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM_LIMIT_RATIO'] = \
            bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']
        bureau['NEW_BURO_AMT_CREDIT_SUM_AMT_CREDIT_SUM_LIMIT_RATIO'] = \
            bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_LIMIT']
        bureau['NEW_BURO_AMT_CREDIT_SUM_AMT_CREDIT_SUM_OVERDUE_PROD'] = \
            bureau['AMT_CREDIT_SUM'] * bureau['AMT_CREDIT_SUM_OVERDUE']
        bureau['NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_RATIO'] = \
            bureau['AMT_ANNUITY'] / (bureau['AMT_CREDIT_SUM'] + 1)
        bureau['NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_DEBT_RATIO'] = \
            bureau['AMT_ANNUITY'] / (bureau['AMT_CREDIT_SUM_DEBT'] + 1)
        bureau['NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_LIMIT_RATIO'] = \
            bureau['AMT_ANNUITY'] / (bureau['AMT_CREDIT_SUM_LIMIT'] + 1)
        bureau['NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_CREDIT_DIFF'] = \
            bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT']
        bureau['NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_CREDIT_ENDDATE_DIFF'] = \
            bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']
        bureau['NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_ENDDATE_FACT_DIFF'] = \
            bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_ENDDATE_FACT']

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['max', 'mean', 'min', 'var', 'size'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_ENDDATE_FACT': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean', 'min'],
            'CNT_CREDIT_PROLONG': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'min', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'min', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'min', 'sum', 'var'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', ],
            'AMT_ANNUITY': ['max', 'mean', 'min'],
            'DAYS_CREDIT_UPDATE': ['max', 'mean', 'min'],
            'MONTHS_BALANCE_MIN': ['max', 'min'],
            'MONTHS_BALANCE_MAX': ['max', 'min'],
            'MONTHS_BALANCE_SIZE': ['max', 'mean', 'min', 'sum'],
            'NEW_BURO_DAYS_CREDIT_DAYS_CREDIT_ENDDATE_DIFF': ['max', 'mean', 'min', 'var'],
            'NEW_BURO_DAYS_CREDIT_DAYS_ENDDATE_FACT_DIFF': ['max', 'mean', 'min', 'var'],
            'NEW_BURO_DAYS_CREDIT_DAYS_CREDIT_ENDDATE_DIFF': ['max', 'mean', 'min', 'var'],
            'NEW_BURO_AMT_CREDIT_MAX_OVERDUE_CREDIT_DAY_OVERDUE_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_CNT_CREDIT_PROLONG_CREDIT_DAY_OVERDUE_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_MAX_OVERDUE_CNT_CREDIT_PROLONG_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_CREDIT_DAY_OVERDUE_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_CNT_CREDIT_PROLONG_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_DEBT_AMT_CREDIT_SUM_LIMIT_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_AMT_CREDIT_SUM_LIMIT_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_CREDIT_SUM_AMT_CREDIT_SUM_OVERDUE_PROD': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_DEBT_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_AMT_ANNUITY_AMT_CREDIT_SUM_LIMIT_RATIO': ['max', 'mean', 'min'],
            'NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_CREDIT_DIFF': ['max', 'mean', 'min', 'var'],
            'NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_CREDIT_ENDDATE_DIFF': ['max', 'mean', 'min', 'var'],
            'NEW_BURO_DAYS_CREDIT_UPDATE_DAYS_ENDDATE_FACT_DIFF': ['max', 'mean', 'min', 'var'],
        }

        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat:
            cat_aggregations[cat] = ['mean']
        #for cat in bb_cat:
        #    cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(
            {**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(
            ['BURO_' + e[0] + "_" + e[1].upper()
                for e in bureau_agg.columns.tolist()])
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

    def fe_application_prev_before(self, df):
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

        # ===============================
        # 欠損値埋め
        # ===============================
        # Days 365.243 values -> nan
        #df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        #df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        #df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        #df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        #df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

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
#        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
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

    # Preprocess credit_card_balance.csv
    def fe_credit_card_balance_before(self, df):
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
#        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count credit card lines
        df_agg['CC_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg

    # Preprocess POS_CASH_balance.csv
    def fe_pos_cash_before(self, df):
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
#        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count df cash accounts
        df_agg['POS_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg

    # Preprocess installments_payments.csv
    def fe_installments_payments_before(self, df):
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
#        df_agg = df_agg.join(df_agg_pref, how='left', on='SK_ID_CURR')
        # Count dftallments accounts
        df_agg['INSTAL_COUNT'] = df.groupby('SK_ID_CURR').size()
        del df
        gc.collect()
        return df_agg


