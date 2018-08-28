import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
from logging import getLogger
import argparse
import gc

from utils.my_logging import logInit
from data_processing.data_io import DataIO
from data_processing.spec_preprocessor import HomeCreditPreprocessor
from models.my_mlp import myMLPClassifier

from scipy.special import erfinv

import preprocess
#import additional_preprocessing

np.random.seed(100)
plt.switch_backend('agg')

drop_cols = [
#        'index',
#        'Unnamed: 0',
#        'NEW_PREV_DAYS_TERMINATION_MAX_DAYS_EMPLOYED_DIFF',
#        'NEW_PREV_INSTAL_PREV_DAYS_ENTRY_PAYMENT_MAX_MAX_DAYS_EMPLOYED_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MAX_DAYS_EMPLOYED_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MAX_DAYS_BIRTH_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MAX_DAYS_LAST_PHONE_CHANGE_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MAX_DAYS_ID_PUBLISH_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MIN_DAYS_BIRTH_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MIN_DAYS_LAST_PHONE_CHANGE_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MIN_DAYS_DAYS_REGISTRATION_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MIN_OWN_CAR_AGE_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MEAN_DAYS_EMPLOYED_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MEAN_DAYS_LAST_PHONE_CHANGE_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MEAN_DAYS_DAYS_REGISTRATION_DIFF',
#        'NEW_BURO_DAYS_CREDIT_MEAN_OWN_CAR_AGE_DIFF',
#        'NEW_AMT_ANNUITY_POPRAT',
#        'NEW_AMT_INCOME_TOTAL_POPRAT',
#        'NEW_AMT_GOODS_PRICE_POPRAT',
#        'NEW_AMT_CREDIT_POPRAT',
        ]

best_features = pd.read_csv('../importances/importance_2018-08-25-06-46-12.csv')
#best_features = pd.read_csv('../importances/importance_2018-08-25-01-54-45.csv')

#best_features = pd.read_csv('../importances/importance_2018-08-07-05-40-26.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-31-23-50-01.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-31-05-16-51.csv')

#best_features = pd.read_csv('../importances/importance_2018-07-29-13-03-22.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-28-05-53-25.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-28-04-21-33.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-28-02-05-07.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-28-00-27-32.csv')
#best_features = pd.read_csv('../importances/importance_2018-07-27-08-49-50.csv')
#drop_cols += best_features.iloc[:400].sort_values('importance_RAT', ascending=False).feature.head(50).tolist()
#drop_cols += best_features.sort_values('importance_RAT', ascending=False).feature.head(1000).tolist()
#drop_cols += best_features.sort_values('importance_RAT', ascending=False).feature.head(1550).tolist()
#drop_cols += best_features.sort_values('importance_MEAN', ascending=True).feature.head(2500).tolist()
#drop_cols += best_features[best_features.importance_RAT.isnull()].feature.tolist()


id_start = 370000
id_end = 430000


def remove_train_only_category(train_df, test_df):
    for column in tqdm(train_df.columns.values):
        # minmax normalization for continuous data
        if train_df[column].dtype == 'object':
            test_set = set(test_df[column].unique())
            train_df = train_df[train_df[column].isin(test_set)]
    return train_df


# Display/plot feature importance
def display_importances(feature_importance_df_, filename='importance_application'):
#    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    csv_df = feature_importance_df_[["feature", "importance"]].groupby("feature").agg({'importance': ['mean', 'std']})
    csv_df.columns = pd.Index(
        [e[0] + "_" + e[1].upper()
            for e in csv_df.columns.tolist()])
    csv_df['importance_RAT'] = csv_df['importance_STD'] / csv_df['importance_MEAN']
    csv_df.sort_values(by="importance_MEAN", ascending=False).to_csv(filename + '.csv')
#    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
#    plt.figure(figsize=(8, 10))
#    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
#    plt.title('LightGBM Features (avg over folds)')
#    plt.tight_layout()
#    plt.savefig(filename + '.png')


def main():
    logger = getLogger(__name__)
    logInit(logger, log_filename='train.py.log')
    logger.info('start')

    dataio = DataIO(logger=logger)
    prep = HomeCreditPreprocessor(logger=logger)

#    dfs_dict = dataio.read_csvs({
#        'train': '../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550_ins-12mon_500.csv',
#        'test': '../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550_ins-12mon_500.csv'})
#        'train': '../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550.csv',
#        'test': '../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550.csv'})
#        'train': '../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000.csv',
#        'test': '../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000.csv'})
#        'train': '../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1300.csv',
#        'test': '../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1300.csv'})
#        'train': '../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1500.csv',
#        'test': '../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1500.csv'})
#        'train': '../inputs/my_train_all.csv',
#        'test': '../inputs/my_test_all.csv'})

#    source_train_df = prep.onehot_encoding(dfs_dict['train'])
#    test_df = prep.onehot_encoding(dfs_dict['test'])
#    train_df = dfs_dict['train']
#    INVALID_IDS = [141289, 144669, 196708, 319880]
#    train_df = train_df[~train_df.SK_ID_CURR.isin(INVALID_IDS)]
#    test_df = dfs_dict['test']

    train_df = pd.read_feather('../inputs/my_train_all.fth')
    test_df = pd.read_feather('../inputs/my_test_all.fth')

#    train_df = train_df.merge(pd.read_csv('../inputs/my_train_all_additional.csv'), on='SK_ID_CURR', how='left')
#    test_df = test_df.merge(pd.read_csv('../inputs/my_test_all_additional.csv'), on='SK_ID_CURR', how='left')

#    _train_df, _test_df = additional_preprocessing.main()
#    train_df, test_df = preprocess.main()
#    _train_df, _test_df = preprocess.main()
#    train_df = train_df.merge(_train_df, on='SK_ID_CURR', how='left')
#    test_df = test_df.merge(_test_df, on='SK_ID_CURR', how='left')
#    del _train_df, _test_df
#    gc.collect()

    logger.info('removing the categorical features which\
                are contained only by training set...')
#    train_df = remove_train_only_category(train_df, test_df)
    train_and_test_df = pd.concat([train_df, test_df], axis=0)
    train_and_test_df, _ = prep.onehot_encoding(train_and_test_df)
#    train_and_test_df['NEW_EXT_SOURCES_MEAN'] = train_and_test_df[['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    train_and_test_df = train_and_test_df.drop(drop_cols, axis=1)
#    train_and_test_df = train_and_test_df.merge(pd.read_csv('../inputs/ext_sources_4_2018-07-31-10-42-54.csv'), on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(pd.read_csv('../inputs/ext_sources_4_2018-07-30-04-46-46.csv'), on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(pd.read_csv('../inputs/ext_sources_4_2018-07-30-02-38-58.csv'), on='SK_ID_CURR', how='left')
#    train_and_test_df = train_and_test_df.merge(pd.read_csv('../inputs/ext_sources_4_2018-07-29-11-28-13.csv'), on='SK_ID_CURR', how='left')

    train_df = train_and_test_df.iloc[:train_df.shape[0]].reset_index(drop=True)
    #train_df = train_and_test_df.iloc[:train_df.shape[0] - 4]
    test_df = train_and_test_df.iloc[train_df.shape[0]:]

    # 分布がおかしいところを upsampling
    up_train_df_index = train_df[(train_df.SK_ID_CURR > 370000) & (train_df.SK_ID_CURR < 430000)].loc
#    up_train_df = train_df[(train_df.SK_ID_CURR > 370000) & (train_df.SK_ID_CURR < 430000)].
#    train_df = pd.concat([train_df] + [up_train_df for i in range(5)], axis=0)

#    train_df.to_csv('../inputs/my_train_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550_ins-12mon_500.csv')
#    test_df.to_csv('../inputs/my_test_all_LGBMClassifier_auc-0.796075_2018-07-28-00-27-32_1000_550_ins-12mon_500.csv')
#    train_df.to_feather('../inputs/my_train_all_LGBMClassifier_auc-0.797984_2018-08-25-08-31-33_drop1000.fth')
#    test_df.to_feather('../inputs/my_test_all_LGBMClassifier_auc-0.797984_2018-08-25-08-31-33_drop1000.fth')
#    exit(0)

    logger.info('encoded training shape is {}'.format(train_df.shape))
    logger.info('encoded test shape is {}'.format(test_df.shape))
#    n_splits = 7
    n_splits = 5
#    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=57)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
#    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)

#    x_train = train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1).values
#    train_df = train_df[(train_df.SK_ID_CURR < 360000) | (train_df.SK_ID_CURR > 430000)]
#    train_df = train_df[(train_df.SK_ID_CURR < 401000) | (train_df.SK_ID_CURR > 415000)]
#    train_df = train_df[(train_df.SK_ID_CURR < 370000)]
#    train_df = train_df[(train_df.SK_ID_CURR < 390000) | (train_df.SK_ID_CURR > 410000)]
#    train_df = train_df[(train_df.SK_ID_CURR < 370000) | ((train_df.SK_ID_CURR > 370000) & (train_df.SK_ID_CURR < 410000)) | (train_df.SK_ID_CURR > 430000)]
    
#    train_and_test_df.iloc[:train_df.shape[0]].TARGET = 0
#    train_and_test_df.iloc[:train_df.shape[0]].TARGET = train_and_test_df.iloc[:train_df.shape[0]].TARGET.fillna(0)
#    train_and_test_df.iloc[train_df.shape[0]:].TARGET = 1
#    train_and_test_df.iloc[train_df.shape[0]:].TARGET = train_and_test_df.iloc[train_df.shape[0]:].TARGET.fillna(1)
#    print(train_and_test_df[train_and_test_df.TARGET.isnull()].TARGET)
#    train_df = train_and_test_df
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
        'nthread': [16],
#        'boosting': ['gbdt', 'gbrt', 'rf', 
#            'random_forest', 'dart', 'goss'],
#        'is_unbalance':[True],
        'n_estimators': [10000],
#        'learning_rates': [lambda iter: 0.1 * (0.995 ** iter)],
        'learning_rate': [0.02],
#        'max_bin': [100],
#        'min_data_in_bin': [50],
##        'num_leaves': [32],
#        'num_leaves': [24],
#        'num_leaves': [24],
        'num_leaves': [31],
#        'num_leaves': [15],
#        'num_leaves': [8],
#        'num_leaves': [48],
        'colsample_bytree': [0.9497036],
#        'subsample': [0.8715623],
        'max_depth': [8],
#        'max_depth': [5],
#        'max_depth': [4],
#        'max_depth': [16],
#        'subsample_freq': [1],
#        'reg_alpha': [0.04],
        'reg_alpha': [0.041545473],
        'reg_lambda': [0.073],
        'min_split_gain': [0.0222415],
        'min_child_weight': [60],
##        'min_child_weight': [40],
        'silent': [-1],
        'verbose': [-1],
        # scale_pos_weight=11<Paste>
    }

    max_score = -1
    best_params = None
    trained_model_ids = {}
#    num_epochs = []
    i = 0
#    import pickle
#    fold_train_test = pickle.load(open('shared_kfold/fold_train_test.pkl', 'rb'))

    for params in tqdm(list(ParameterGrid(all_params))):
        feature_importance_df = pd.DataFrame()
        logger.info('params: {}'.format(params))
        list_score = []
        for trn_idx, val_idx in tqdm(list(skf.split(x_train, y_train))):
        #for i in tqdm(list(range(5))):
            #trn_idx = fold_train_test[i]['train_index']
            #val_idx = fold_train_test[i]['test_index']
            x_trn, x_val = x_train[trn_idx], x_train[val_idx]
            y_trn, y_val = y_train[trn_idx], y_train[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(x_trn, y_trn, eval_set=[(x_trn, y_trn), (x_val, y_val)],
                    eval_metric='auc', verbose=100, early_stopping_rounds=500)

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
    #backend.clear_session()
