import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean', 'median']:
            if not all(np.isin(X.dtypes,
                               [np.dtype('int32'), np.dtype('int64'),
                                   np.dtype('float32'), np.dtype('float64'),
                                   np.dtype('uint8')])):
                raise ValueError('''dtypes mismatch np.number dtype is \
                                 required for {}. your dtypes are {}.\
                                 '''.format(self.strategy, X.dtypes))
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'most_frequent':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'min':
            self.fill = X.min()
        elif self.strategy == 'max':
            self.fill = X.max()
        elif self.strategy == 'fill':
            if isinstance(self.fill, list) and isinstance(X, pd.DataFrame):
                self.fill = dict([(cname, v)
                                  for cname, v in zip(X.columns, self.fill)])
        else:
            raise ValueError("you have to specify 'mean', 'median', \
                    'most_frequent', or 'fill'.")
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
