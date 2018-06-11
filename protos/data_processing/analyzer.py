import pandas as pd
import numpy as np

from tqdm import tqdm

from logging import getLogger


class Analyzer:
    r'''
    the object which analyses the data.

    '''
    def __init__(self, logger=None):
        self.logger = getLogger(logger.name).getChild(
                self.__class__.__name__) if logger else None

    def get_null_stat(self, target_df, sort_target=False, ascending=False):
        r'''
        return null count and percentge for each columns of target_df

        Parameters
        --------------------
        target_df : DataFrame
            the dataframe about which you want to get null stat
        sort_target : string
            the column name based on which you want to sort the result.
        ascending : bool
            the flag which specifes if you want to sort in acsending order.
            you can use this only when you specify sort_target.

        Results
        --------------------
        res_df : DataFrame
            the dataframe which contains the null stats of target_df

        '''
        if type(sort_target) != str:
            raise TypeError('arg sort_target should be str.')

        total_count = target_df.shape[0]
        null_count = target_df.isnull().sum().values
        null_ratio = null_count / total_count
        if sort_target:
            res_df = pd.DataFrame(data={'null_count': null_count,
                                        'null_ratio': null_ratio},
                                  index=target_df.columns.values).sort_values(
                                        sort_target, ascending=ascending)
        else:
            res_df = pd.DataFrame(data={'null_count': null_count,
                                        'null_ratio': null_ratio},
                                  index=target_df.columns.values)
        return res_df
