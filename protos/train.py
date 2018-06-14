import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
from logging import getLogger
import argparse

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor


from scipy.special import erfinv


class GaussRankScaler():

    def __init__(self):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform(self, X):

        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed


# def get_args():
#     return args


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
    logger.info('normalizing inputs...')
#    scaler = MinMaxScaler()
    scaler = GaussRankScaler()
    for column in tqdm(train_and_test_df.columns.values):
        # minmax normalization for continuous data
        if train_and_test_df[column].dtype != 'object'\
                and column != 'SK_ID_CURR'\
                and train_and_test_df[column].nunique() > 100:
#            train_and_test_df[column] = scaler.fit_transform(train_and_test_df[column])
            if train_and_test_df[column].max() > 0:
                train_and_test_df[column] = scaler.fit_transform(train_and_test_df[column])
                #train_and_test_df[column] /= train_and_test_df[column].max()
    train_and_test_df = prep.onehot_encoding(train_and_test_df)
    train_df = train_and_test_df.iloc[:dfs_dict['train'].shape[0]]
    test_df = train_and_test_df.iloc[dfs_dict['train'].shape[0]:]
#    test_size = 0.3
#    train_df, val_df = prep.foldout(source_train_df, test_size)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

#    x_trn = train_df.drop(['TARGET'], axis=1).values
#    x_val = val_df.drop(['TARGET'], axis=1).values
#    x_train = np.concatenate([x_trn, x_val])
    x_train = train_df.drop(['TARGET'], axis=1).values
#    y_trn = train_df['TARGET'].values
#    y_val = val_df['TARGET'].values
#    y_train = np.concatenate([y_trn, y_val])
    y_train = train_df['TARGET'].values
    x_test = test_df.drop(['TARGET'], axis=1).values

    all_params = {
        'max_iter': [100],
        'solver': ['adam'],
        'hidden_layer_sizes': [(1000, 1000, )],
        'verbose': [True],
        'early_stopping': [True],
        'validation_fraction': [0.3],
        'random_state': [0],
        'learning_rate_init': [0.0001, 0.00001],
#        'alpha': [0.01, 0.0001],
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

    max_score = -1
    best_params = None
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))
        for trn_idx, val_idx in tqdm(list(skf.split(x_train, y_train))):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

            list_score = []

            clf = MLPClassifier(**params)
            clf.fit(x_trn, y_trn)

            pred_prob = clf.predict_proba(x_val)[:, 1]
            auc_score = roc_auc_score(y_val, pred_prob)
            logger.debug('auc : {}'.format(auc_score))
            list_score.append(auc_score)
            break

        auc_score = np.array(list_score).mean()
        logger.info('avg auc score of the current cv : {}'.format(auc_score))
        list_score = []
        if max_score < auc_score:
            max_score = auc_score
            best_params = params
        logger.info('model: {}'.format(clf.__class__.__name__))
        logger.info('current max score: {}'.format(max_score))
        logger.info('current best params: {}'.format(best_params))
#        break

    logger.info('model: {}'.format(clf.__class__.__name__))
    logger.info('max score: {}'.format(max_score))
    logger.info('best params: {}'.format(best_params))
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

    dataio.save_csv(res_df, '../submits/{}_auc-{}.csv'.format(
        clf.__class__.__name__, max_score), index=False, withtime=True)

    logger.info('end')


if __name__ == '__main__':
    main()
