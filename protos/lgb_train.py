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

#from sklearn.category_encoders.targetencoder import TargetEncoder

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
import argparse

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor
from models.my_mlp import myMLPClassifier


from scipy.special import erfinv

np.random.seed(100)
plt.switch_backend('agg')


def remove_train_only_category(train_df, test_df):
    for column in tqdm(train_df.columns.values):
        # minmax normalization for continuous data
        if train_df[column].dtype == 'object':
            test_set = set(test_df[column].unique())
            train_df = train_df[train_df[column].isin(test_set)]
    return train_df


# Display/plot feature importance
def display_importances(feature_importance_df_, filename='importance_application'):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).to_csv(filename + '.csv')
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(filename + '.png')


def main():
    logger = getLogger(__name__)
    logInit(logger, log_filename='train.py.log')
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

    dfs_dict = dataio.read_csvs({
        'train': '../inputs/my_train.csv',
        'test': '../inputs/my_test.csv'})

#    source_train_df = prep.onehot_encoding(dfs_dict['train'])
#    test_df = prep.onehot_encoding(dfs_dict['test'])
#    dfs_dict['train'] = prep.down_sampling(dfs_dict['train'], 'TARGET')
    train_df = dfs_dict['train']
    test_df = dfs_dict['test']
    logger.info('removing the categorical features which\
                are contained only by training set...')
#    train_df = remove_train_only_category(train_df, test_df)
    train_and_test_df = pd.concat([train_df, test_df], axis=0)
    train_and_test_df, _ = prep.onehot_encoding(train_and_test_df)

#    importance_list = pd.read_csv('../importances/importance_2018-07-11-08-58-40.csv')
#    train_and_test_df = train_and_test_df[importance_list.feature[:150].tolist() + ['SK_ID_CURR', 'TARGET']]

    train_df = train_and_test_df.iloc[:train_df.shape[0]]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]
    logger.info('encoded training shape is {}'.format(train_df.shape))
    logger.info('encoded test shape is {}'.format(test_df.shape))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
#    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

#    x_train = train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values
    x_train = train_df.drop([
        'TARGET', 
        'SK_ID_CURR', 
#        "AMT_CREDIT", 
#        "AMT_REQ_CREDIT_BUREAU_DAY", 
#        "NEW_EXT_SOURCES_MEAN", 
#        "PREV_CODE_REJECT_REASON_LIMIT_MEAN"
        ], axis=1)
    train_feats = x_train.columns
    x_train = x_train.values

#    x_train = train_df.drop([
#        'TARGET', 'SK_ID_CURR',
#        'EXT_SOURCE_1', 'EXT_SOURCE_2',
#        'EXT_SOURCE_3'], axis=1).values
    y_train = train_df['TARGET'].values
    x_test = test_df.drop([
        'TARGET', 
        'SK_ID_CURR',
#        "AMT_CREDIT", 
#        "AMT_REQ_CREDIT_BUREAU_DAY", 
#        "NEW_EXT_SOURCES_MEAN", 
#        "PREV_CODE_REJECT_REASON_LIMIT_MEAN",
        ], axis=1).values

    all_params = {
        'nthread': [-1],
#        'boosting': ['gbdt', 'gbrt', 'rf', 
#            'random_forest', 'dart', 'goss'],
        # is_unbalance=True,
#        'boosting_type': ['goss'],
        'n_estimators': [10000],
        'learning_rate': [0.02],
#        'learning_rate': [0.02],
        'num_leaves': [32],
        'colsample_bytree': [0.9497036],
        'subsample': [0.8715623],
        'max_depth': [16],
#        'max_depth': [16],
#        'reg_alpha': [0.04],
#        'reg_lambda': [0.073],
#        'reg_lambda': [0.0, 0.1, 0.2],
        'min_split_gain': [0.0222415],
        'min_child_weight': [40],
        'silent': [-1],
        'verbose': [-1],
        # scale_pos_weight=11<Paste>
    }

    max_score = -1
    best_params = None
    trained_model_ids = {}
#    num_epochs = []
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
                    eval_metric='auc', verbose=100, early_stopping_rounds=200)

            pred_prob = clf.predict_proba(
                    x_val, num_iteration=clf.best_iteration_)[:, 1]
            auc_score = roc_auc_score(y_val, pred_prob)
            logger.debug('auc : {}'.format(auc_score))
            list_score.append(auc_score)
            if i in trained_model_ids:
                trained_model_ids[i].append(clf)
            else:
                trained_model_ids[i] = [clf, ]
#            if len(trained_model_ids[i]) > 1:
#                break
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

    dataio.save_csv(res_df, '../submits/{}_auc-{:.6}.csv'.format(
        clf.__class__.__name__, max_score), index=False, withtime=True)

    logger.info('end')


if __name__ == '__main__':
    main()
    # Exception ignored in: <bound method BaseSession.__del__ of
    # <tensorflow.python.client.session.Session object at 0x1248ba668>>
    # の対策
    backend.clear_session()
