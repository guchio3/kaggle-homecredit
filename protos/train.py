import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
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
         device_count={'CPU': 16})))

from tqdm import tqdm
from logging import getLogger
import argparse

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor
from models.my_mlp import myMLPClassifier


from scipy.special import erfinv

np.random.seed(100)


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


def continuousNormalization(target_df, scaler):
    for column in tqdm(target_df.columns.values):
        # minmax normalization for continuous data
        if target_df[column].dtype != 'object'\
                and column != 'SK_ID_CURR'\
                and target_df[column].nunique() > 100:
            if target_df[column].max() > 0:
                target_df[column] = \
                    scaler.fit_transform(target_df[column])
    return target_df


def remove_train_only_category(train_df, test_df):
    for column in tqdm(train_df.columns.values):
        # minmax normalization for continuous data
        if train_df[column].dtype == 'object':
            test_set = set(test_df[column].unique())
            train_df = train_df[train_df[column].isin(test_set)]
    return train_df


def main():
    logger = getLogger(__name__)
    logInit(logger, log_filename='train.py.log')
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

#    dfs_dict = dataio.read_csvs({
#        'train': '../inputs/my_train_2.csv',
#        'test': '../inputs/my_test_2.csv'})

#    source_train_df = prep.onehot_encoding(dfs_dict['train'])
#    test_df = prep.onehot_encoding(dfs_dict['test'])
#    dfs_dict['train'] = prep.down_sampling(dfs_dict['train'], 'TARGET')
#    train_df = dfs_dict['train']
#    test_df = dfs_dict['test']
    train_df = pd.read_feather('../inputs/my_train_all.fth')
    test_df = pd.read_feather('../inputs/my_test_all.fth')

    logger.info('removing the categorical features which\
                are contained only by training set...')
#    train_df = remove_train_only_category(train_df, test_df)
    train_and_test_df = pd.concat([train_df, test_df], axis=0)
    logger.info('imputing...')
    train_and_test_df = prep.auto_impute(train_and_test_df)
    logger.info('normalizing inputs...')
#    scaler = MinMaxScaler()
    scaler = GaussRankScaler()
    # apply gaussrank normalization
    train_and_test_df = continuousNormalization(train_and_test_df, scaler)
    train_and_test_df, _ = prep.onehot_encoding(train_and_test_df)
    train_df = train_and_test_df.iloc[:train_df.shape[0]]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]
    logger.info('encoded training shape is {}'.format(train_df.shape))
    logger.info('encoded test shape is {}'.format(test_df.shape))
    n_splits = 5
    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)

#    x_train = train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values
    x_train = train_df.drop([
        'TARGET', 'SK_ID_CURR'], axis=1).values
#    x_train = train_df.drop([
#        'TARGET', 'SK_ID_CURR',
#        'EXT_SOURCE_1', 'EXT_SOURCE_2',
#        'EXT_SOURCE_3'], axis=1).values
    y_train = train_df['TARGET'].values
    x_test = test_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values

    all_params = {
        'max_iter': [200],
        'solver': ['sgd'],
        #'solver': ['sgd', 'adam'],
#        'hidden_layer_sizes': [(250, 70), ],
        'hidden_layer_sizes': [(400, 40)],
        #'hidden_layer_sizes': [(150, 35), (300, 50), (500, 100), (500, 50), (400, 40)],
        #'hidden_layer_sizes': [(150, 35)],
#        'hidden_layer_sizes': [(100, 30), (150, 35)],
#        'hidden_layer_sizes': [(30, ), (30, 30, 30, ), (100, 30)],
        'random_state': [0, 17, 41, 333, 1001],
#        'learning_rate_init': [0.00001, ],
        'learning_rate_init': [0.7, ],
        'alpha': [0.05, ],
        'batch_size': [256],
        'batch_norm': [(False, True), ],
        'dropout': [(0.05, 0.5,)],
#        'dropout': [(0.1, 0.3), (0.1, 0.5)],
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
    best_models = -1
    trained_model_ids = {}
#    num_epochs = []
    i = 0
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))
        list_score = []
        for trn_idx, val_idx in tqdm(list(skf.split(x_train, y_train))):
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

#            clf = MLPClassifier(**params)
            clf = myMLPClassifier(logger=logger, **params)
            #if clf.__call__.__name__ == 'myMLPClassifier':
#            clf.fit(x_trn, y_trn)
            clf.fit(x_trn, y_trn, 
                    eval_set=[x_val, y_val],
                    best_model_filename='temp_model.h5')
            clf.load_weights(load_filename='temp_model.h5')
            #num_epochs.append(len(r.history['loss']))

            pred_prob = clf.predict_proba(x_val)[:, 1]
            #pred_prob = clf.predict_proba(x_val)[:, 0]
            auc_score = roc_auc_score(y_val, pred_prob)
            logger.debug('auc : {}'.format(auc_score))
            list_score.append(auc_score)
            if i in trained_model_ids:
                trained_model_ids[i].append(clf)
            else:
                trained_model_ids[i] = [clf, ]
#            if len(trained_model_ids[i]) > 1:
#                break

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

    logger.info('model: {}'.format(clf.__class__.__name__))
    logger.info('max score: {}'.format(max_score))
    logger.info('best params: {}'.format(best_params))
    logger.info('start re-training')
#    best_params['max_iter'] = np.mean() * (n_splits / (n_splits - 1))
    for clf in trained_model_ids[best_model_id]:
        continue
        clf.fit()
#    clf = MLPClassifier(**best_params)
#    clf = myMLPClassifier(logger=logger, **best_params)
#    clf = XGBClassifier(**best_params)
    # clf = LogisticRegression(**best_params)
#    clf.fit(x_train, y_train)

    logger.info('train end')

#    x_test = sel.transform(x_test)
#    res = clf.predict_proba(x_test)[:, 1]

    reses = []
#    for clf in trained_model_ids[best_model_id]:
#        reses.append(clf.predict_proba(x_test)[:, 1])
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
        'TARGET': res
    })

    dataio.save_csv(res_df, '../submits/{}_auc-{:.4}.csv'.format(
        clf.__class__.__name__, max_score), index=False, withtime=True)

    logger.info('end')


if __name__ == '__main__':
    main()
    # Exception ignored in: <bound method BaseSession.__del__ of
    # <tensorflow.python.client.session.Session object at 0x1248ba668>>
    # の対策
    backend.clear_session()
