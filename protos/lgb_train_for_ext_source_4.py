import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
from logging import getLogger
import argparse

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor
from models.my_mlp import myMLPClassifier

from scipy.special import erfinv

import preprocess

np.random.seed(100)
plt.switch_backend('agg')

# Display/plot feature importance
def display_importances(feature_importance_df_, filename='importance_application'):
    csv_df = feature_importance_df_[["feature", "importance"]].groupby("feature").agg({'importance': ['mean', 'std']})
    csv_df.columns = pd.Index(
        [e[0] + "_" + e[1].upper()
            for e in csv_df.columns.tolist()])
    csv_df['importance_RAT'] = csv_df['importance_STD'] / csv_df['importance_MEAN']
    csv_df.sort_values(by="importance_MEAN", ascending=False).to_csv(filename + '.csv')


def main():
    logger = getLogger(__name__)
    logInit(logger, log_filename='train.py.log')
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)
    cols = [
            'SK_ID_CURR',
            'AMT_ANNUITY',
            'AMT_CREDIT',
            'AMT_GOODS_PRICE',
            'HOUR_APPR_PROCESS_START',
            'NAME_CONTRACT_TYPE',
            'NAME_TYPE_SUITE',
            'WEEKDAY_APPR_PROCESS_START',
            ]

    dfs_dict = dataio.read_csvs({
        'prev': '../inputs/previous_application.csv',
        'train': '../inputs/application_train.csv',
        'test': '../inputs/application_test.csv'})

    prev_df = dfs_dict['prev'][cols + ['SK_ID_PREV']]
    train_df = dfs_dict['train'][cols]
    test_df = dfs_dict['test'][cols]

    logger.info('removing the categorical features which\
                are contained only by training set...')

    ins_df = pd.read_csv('../inputs/installments_payments.csv').sort_values(['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
    ins_df['DIFF'] = ins_df.DAYS_ENTRY_PAYMENT - ins_df.DAYS_INSTALMENT
#    ins_df = ins_df.groupby('SK_ID_PREV').head(3).groupby('SK_ID_PREV').DIFF.max().reset_index()
    ins_df = ins_df.groupby('SK_ID_PREV').head(12).groupby('SK_ID_PREV').DIFF.max().reset_index()

#    bb_df = pd.read_csv('../inputs/bureau_balance.csv')
#    bureau_df = pd.read_csv('../inputs/bureau.csv')
#    bureau_df = prep.fe_bureau_and_balance(bureau_df, bb_df)
    base_train_df = pd.read_csv('../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550.csv')
    base_test_df = pd.read_csv('../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550.csv')
    base_df = pd.concat([base_train_df, base_test_df], axis=0).drop(['TARGET'], axis=1)

    train_and_test_df = pd.concat([prev_df, train_df, test_df], axis=0)
    train_and_test_df, _ = prep.onehot_encoding(train_and_test_df)
    train_and_test_df['NEW_CREDIT_TO_ANNUITY_RATIO'] = train_and_test_df['AMT_CREDIT'] / train_and_test_df['AMT_ANNUITY']
    train_and_test_df['NEW_CREDIT_TO_GOODS_RATIO'] = train_and_test_df['AMT_CREDIT'] / train_and_test_df['AMT_GOODS_PRICE']
    train_and_test_df['NEW_ANNUITY_GOODS_TO_RATIO'] = train_and_test_df['AMT_ANNUITY'] / train_and_test_df['AMT_GOODS_PRICE']
    train_and_test_df = train_and_test_df.merge(base_df, on='SK_ID_CURR', how='left')
    del base_df

#    train_and_test_df = train_and_test_df.merge(
#            bureau_df, on='SK_ID_CURR', how='left')

    train_df = train_and_test_df.iloc[:prev_df.shape[0]]
    test_df = train_and_test_df.iloc[prev_df.shape[0]:]
    train_df = train_df.merge(ins_df, on='SK_ID_PREV')
    train_df['TARGET'] = (train_df.DIFF > 10).apply(lambda x: 1 if x else 0)
    #train_df['TARGET'] = (train_df.DIFF > 3).apply(lambda x: 1 if x else 0)
    #train_df['TARGET'] = (train_df.DIFF > 0).apply(lambda x: 1 if x else 0)
    train_df['TARGET'] = train_df['TARGET'].fillna(0)

    logger.info('encoded training shape is {}'.format(train_df.shape))
    logger.info('encoded test shape is {}'.format(test_df.shape))

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)

    x_train = train_df.drop([
        'TARGET', 
        'SK_ID_CURR', 
        'SK_ID_PREV', 
        'DIFF', 
        ], axis=1)
    train_feats = x_train.columns
    x_train = x_train.values

    y_train = train_df['TARGET'].values
    x_test = test_df.drop([
        'SK_ID_CURR',
        'SK_ID_PREV',
        ], axis=1).values

    all_params = {
        'nthread': [-1],
        'n_estimators': [10000],
        'learning_rate': [0.2],
        'num_leaves': [8],
        'colsample_bytree': [0.9497036],
        'subsample': [0.8715623],
        'max_depth': [4],
#        'reg_alpha': [0.04],
#        'reg_lambda': [0.073],
        'min_split_gain': [0.0222415],
        'min_child_weight': [60],
        'silent': [-1],
        'verbose': [-1],
    }

    max_score = -1
    best_params = None
    trained_model_ids = {}

    i = 0
    for params in tqdm(list(ParameterGrid(all_params))):
        feature_importance_df = pd.DataFrame()
        logger.info('params: {}'.format(params))
        list_score = []
        for trn_idx, val_idx in tqdm(list(skf.split(x_train, y_train))):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(x_trn, y_trn, eval_set=[(x_trn, y_trn), (x_val, y_val)],
                    eval_metric='auc', verbose=100, early_stopping_rounds=300)

            pred_prob = clf.predict_proba(
                    x_val, num_iteration=clf.best_iteration_)[:, 1]
            auc_score = roc_auc_score(y_val, pred_prob)
            logger.debug('auc : {}'.format(auc_score))
            list_score.append(auc_score)
            if i in trained_model_ids:
                trained_model_ids[i].append(clf)
            else:
                trained_model_ids[i] = [clf, ]
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = train_feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = i + 1
            feature_importance_df = pd.concat(
                    [feature_importance_df, fold_importance_df], axis=0)

        auc_score = np.array(list_score).mean()
        logger.info('avg auc score of the current cv : {}'.format(auc_score))
        if max_score < auc_score:
            max_score = auc_score
            best_params = params
            best_model_id = i
        logger.info('model: {}'.format(clf.__class__.__name__))
        logger.info('current max score: {}'.format(max_score))
        logger.info('current best params: {}'.format(best_params))
        i += 1
#        break

    logger.info('displaying feature importance')
    display_importances(feature_importance_df, '../importances/importance_{}'.format(dataio.current_time))
    logger.info('model: {}'.format(clf.__class__.__name__))
    logger.info('max score: {}'.format(max_score))
    logger.info('best params: {}'.format(best_params))
    logger.info('start re-training')
    for clf in trained_model_ids[best_model_id]:
        continue
        clf.fit()

    logger.info('train end')

    reses = []
    for key in trained_model_ids.keys():
        for clf in trained_model_ids[key]:
            reses.append(clf.predict_proba(x_test)[:, 1])
    print(reses)
    res = reses[0]
    for r in reses[1:]:
        res += r
    res /= len(reses)
    print(res)

    logger.info('formatting the test result...')
    res_df = pd.DataFrame({
        'SK_ID_CURR': test_df.SK_ID_CURR,
        'EXT_SOURCE_4': res
    })

    dataio.save_csv(res_df, '../inputs/ext_sources_4.csv', index=False, withtime=True)

    logger.info('end')


if __name__ == '__main__':
    main()
