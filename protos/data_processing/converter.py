import numpy as np
import pandas as pd

from tqdm import tqdm

from logging import getLogger


class Converter:
    def __init__(self, logger=None):
        self.logger = getLogger(logger.name).getChild(self.__name__) \
            if logger else None

    def onehot_encoding(self, target_df, additional_features=None):
        r'''
        Encode the categorical columns of the input df into onehot style.
        This function recognizes the columns as categorical ones if the type
                is "object", so if you want to specify the other types of
                columns as categorical, please use additional_features.


        Parameters
        --------------------
        target_df : DataFrame
            the dataframe to which you want to apply onehot encoding.
        additional_features : list of strings
            the names of columns which you want to additinally specify
            as categorical.

        Results
        --------------------
        target_df : DataFrame
            the dataframe which is applied onehot encoding.
        '''
        if self.logger:
            self.logger.info('encoding dataframe as onehot style'.format())
        for col in tqdm(target_df.columns.values):
            if col.dtype == 'object' or col in additional_features:
                if self.logger:
                    self.logger.debug('encoding the category {} \
                        as onehot style...'.format(col))
                tmp = pd.get_dummies(target_df[col], col)
                for col2 in tmp.columns.values:
                    target_df[col2] = tmp[col2].values
                target_df.drop(col, axis=1, inplace=True)
        return target_df
