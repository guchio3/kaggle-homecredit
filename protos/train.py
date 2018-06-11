import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
from logging import getLogger

import collections
import datetime
import json

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor


def main():
    logger = getLogger(__name__)
    logInit(logger)
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

    dfs_dict = dataio.read_csvs({
        'train': '../inputs/my_train.csv',
        'test': '../inputs/my_test.csv'})

#    source_train_df = prep.onehot_encoding(dfs_dict['train'])
#    test_df = prep.onehot_encoding(dfs_dict['test'])
    train_and_test_df = pd.concat([dfs_dict['train'],
                                   dfs_dict['test']], axis=0)
    train_and_test_df = prep.onehot_encoding(train_and_test_df)
    train_df = train_and_test_df.iloc[:dfs_dict['train'].shape[0]]
    test_df = train_and_test_df.iloc[dfs_dict['train'].shape[0]:]

#    test_size = 0.3
#    train_df, val_df = prep.foldout(source_train_df, test_size)

#    x_trn = train_df.drop(['TARGET'], axis=1).values
#    x_val = val_df.drop(['TARGET'], axis=1).values
#    x_train = np.concatenate([x_trn, x_val])
    x_train = train_df.drop(['TARGET'], axis=1).values
#    y_trn = train_df['TARGET'].values
#    y_val = val_df['TARGET'].values
#    y_train = np.concatenate([y_trn, y_val])
    y_train = train_df['TARGET'].values
    x_test = test_df.drop(['TARGET'], axis=1).values

#    print(x_train.T @ y_train)
#    exit(0)
#    print(x_train)
#    print(type(x_train))
#    print(x_train.shape)
#    print(y_train)
#    print(type(y_train))
#    print(y_train.shape)
#    exit(0)

    all_params = {
        'max_iter': [100],
        'solver': ['lbfgs', 'adam'],
        'hidden_layer_sizes': [(20, )],
        'verbose': [True],
        'early_stopping': [True],
        'validation_fraction': [0.3],
        'random_state': [777],
    }

#    all_params = {}

    # logistic regression params
#    all_params = {
#        'max_iter': [500],
#        'solver': ['liblinear'],
#        'multi_class': ['ovr'],
#        'C': [0.1],
#        'penalty': ['l2'],
#        'random_state': [1],
#        'class_weight': ['balanced'],
#        'verbose': [1],
#    }

    for params in tqdm(list(ParameterGrid(all_params))):
        all_params = params
        break
#        logger.debug('params: {}'.format(params))
#        list_score = []
#
#        clf = MLPClassifier(**params)
#        clf.fit(x_trn, y_trn)
#
#        pred = clf.predict(x_val)
#        acc_score = clf.score(x_val, y_val)
#        #pred_prob = clf.predict_proba(x_val)[:, ]
#        score = acc_score
#        logger.debug('pred stat : {}'.format(collections.Counter(pred)))
#        logger.debug('ndcg : {}, acc : {}'.format(score, acc_score))
#
#        validation_file = '../validation_log/validation_df_{}.csv'.format(current_time)
#
#        if max_score < score:
#            max_score = score
#            best_params = params
#        logger.debug('current model: {}'.format(clf.__class__.__name__))
#        logger.debug('current max score: {}, \
#                current best params: {}'.format(max_score, best_params))
#
#        cnt += 1

    best_params = all_params
    logger.info('best params: {}'.format(best_params))
#    logger.info('max score: {}'.format(max_score))

    logger.info('start re-training')
    clf = MLPClassifier(**best_params)
#    clf = XGBClassifier(**best_params)
    # clf = LogisticRegression(**best_params)
    clf.fit(x_train, y_train)

    logger.info('train end')

#    x_test = sel.transform(x_test)
    res = clf.predict_proba(x_test)[:, 1]

    logger.info('formatting the test result...')
    res_df = pd.DataFrame({
        'SK_ID_CURR': test_df.SK_ID_CURR,
        'TARGET': res
    })

#    dataio.save_csv(res_df, '../submits/submit_{}_{}.tsv'.format(
#        clf.__class__.__name__, current_time), index=False)
    dataio.save_csv(res_df, '../submits/sample_submit.csv')

    logger.info('end')


if __name__ == '__main__':
    main()
