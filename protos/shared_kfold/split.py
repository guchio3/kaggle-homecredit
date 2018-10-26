from sklearn.cross_validation import KFold
import numpy as np
def splitter(clf, x_train, y, x_test):
  NFOLDS = 5
  SEED   = 71
  kf = KFold(len(x_train), n_folds=NFOLDS, shuffle=True, random_state=SEED)

  fold_train_test = {}
  for i, (train_index, test_index) in enumerate(kf):
    fold_train_test[i] = { 'train_index': train_index, 'test_index':test_index }
  return fold_train_test

import glob, gc
import pandas as pd
df       = pd.read_pickle('../my-nural/00-dump.pkl')
train_df = df[df['TARGET'].notnull()]
test_df  = df[df['TARGET'].isnull()]
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
y = train_df['TARGET']

# この差は4件あるが、7.95だしているカーネルによるとこれは、正しい
print(train_df.shape)
print(pd.read_csv('../input/application_train.csv').shape)
check = pd.read_pickle('../my-nural/00-dump.pkl')
print( check[check['TARGET'].notnull()].shape )
fold_train_test = splitter(None, train_df,  y.values, test_df)

import pickle

pickle.dump(fold_train_test, open('fold_train_test.pkl', 'wb'))
# pickle形式で保存
# 例) 2foldのtrain_indexにアクセス
# fold_train_test = pickle.load(open('fold_train_test.pkl', 'rb'))

print(fold_train_test[2]['train_index'] )

