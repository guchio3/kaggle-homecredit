import numpy as np
import pandas as pd
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

from keras.utils import plot_model
from keras.backend import tensorflow_backend as backend
from keras import backend as K
K.set_session(K.tf.Session(
    config=K.tf.ConfigProto(
        #        intra_op_parallelism_threads=2,
        #        inter_op_parallelism_threads=2)))
        device_count={'CPU': 8})))

from tqdm import tqdm
from logging import getLogger
import json
import argparse

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor
from models.my_mlp import myMLPClassifier


from scipy.special import erfinv

np.random.seed(100)


def remove_train_only_category(train_df, test_df):
    for column in tqdm(train_df.columns.values):
        # minmax normalization for continuous data
        if train_df[column].dtype == 'object':
            test_set = set(test_df[column].unique())
            train_df = train_df[train_df[column].isin(test_set)]
    return train_df


def main():
    logger = getLogger(__name__)
    logInit(logger, log_filename='grid_feature_search.py.log')
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

    dfs_dict = dataio.read_csvs({
        'train': '../inputs/my_train_2_w_missing_and_was_null.csv',
        'test': '../inputs/my_test_2_w_missing_and_was_null.csv'})

    train_df = dfs_dict['train']
    test_df = dfs_dict['test']
    logger.info('removing the categorical features which\
                are contained only by training set...')
    train_and_test_df = pd.concat([train_df, test_df], axis=0)
    train_and_test_df, _ = prep.onehot_encoding(train_and_test_df)
    train_df = train_and_test_df.iloc[:train_df.shape[0]]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]
    logger.info('encoded training shape is {}'.format(train_df.shape))
    logger.info('encoded test shape is {}'.format(test_df.shape))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

    x_train_df = train_df.drop([
        'TARGET', 'SK_ID_CURR'], axis=1)
#    x_train = train_df.drop([
#        'TARGET', 'SK_ID_CURR',
#        'EXT_SOURCE_1', 'EXT_SOURCE_2',
#        'EXT_SOURCE_3'], axis=1).values
    y_train = train_df['TARGET'].values
    x_test = test_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values

    params = {
#        'nthread': [4],
        # is_unbalance=True,
#        'n_estimators': [10000],
        'n_estimators': 500,
        'learning_rate': 0.02,
        'num_leaves': 32,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'max_depth': 8,
        'reg_alpha': 0.04,
        'reg_lambda': 0.073,
        'min_split_gain': 0.0222415,
        'min_child_weight': 40,
        'silent': -1,
        'verbose': -1,
        # scale_pos_weight=11<Paste>
    }

    max_score = -1
    dropped_features = []
    i = 0
    for feature in tqdm([None] + x_train_df.columns.tolist()):
        logger.info('dropping feature... {}'.format(feature))
        list_score = []
        if feature is not None:
            x_train = x_train_df.drop([feature], axis=1).values
        else:
            x_train = x_train_df.values
        for trn_idx, val_idx in tqdm(list(skf.split(x_train, y_train))):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(x_trn, y_trn, eval_set=[(x_trn, y_trn), (x_val, y_val)],
                    eval_metric='auc', verbose=100)

            pred_prob = clf.predict_proba(x_val)[:, 1]
            auc_score = roc_auc_score(y_val, pred_prob)
            logger.debug('auc : {}'.format(auc_score))
            list_score.append(auc_score)
            break

        auc_score = np.array(list_score).mean()
        logger.info('avg auc score of the current cv : {}'.format(auc_score))
        if i == 0:
            i += 1
            max_score = auc_score
        elif max_score < auc_score:
            logger.info('dropping {} is right'.format(feature))
            max_score = auc_score
            dropped_features.append(feature)
            x_train_df = x_train_df.drop([feature], axis=1)
        logger.info('model: {}'.format(clf.__class__.__name__))
        logger.info('current max score: {}'.format(max_score))

    logger.info('max score: {}'.format(max_score))
    logger.info('dropped features {}'.format(dropped_features))
    try:
        with open('./dropped_features.json', 'w') as fout:
            fout.write(json.dumps(dropped_features))
    except:
        with open('./dropped_features.json', 'wb') as fout:
            fout.write(json.dumps(dropped_features))

    logger.info('end')


if __name__ == '__main__':
    main()
    backend.clear_session()
