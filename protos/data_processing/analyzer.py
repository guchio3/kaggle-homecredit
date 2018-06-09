import pandas as pd
import numpy as np

from tqdm import tqdm

from logging import getLogger


class Analyzer:
    r'''
    the object which analyses the data.

    '''
    def __init__(self, logger=None):
        self.logger = getLogger(logger.name).getChild(self.__name__) \
            if logger else None

    def feature_type_split(self, data, special_list=[], uniq_disc_num_max=100):
        r'''
        a function which categorises the columns of input data to three types,
        namely, cattegorical, discrete numerical, and continuous numerical.

        * Example usage
        >>> cat_list, dis_num_list, num_list = feature_type_split(
                application_train, special_list=['AMT_REQ_CREDIT_BUREAU_YEAR'])
        >>> cat_list
        ['NAME_CONTRACT_TYPE', 'CODE_GENDER', ... ,'EMERGENCYSTATE_MODE']
        >>> dis_num_list
        ['TARGET', 'CNT_CHILDREN', ... ,'AMT_REQ_CREDIT_BUREAU_YEAR']
        >>> num_list
        ['SK_ID_CURR', 'AMT_INCOME_TOTAL', ... ,'DAYS_LAST_PHONE_CHANGE']


        Parameters
        --------------------
        data : DataFrame
            the dataframe about which you want to categorize
        special_list : list of strings
            the list of column names which you specifically use as
            discrete numerical features.
        uniq_disc_num_max : integer
            the number of uniq elements which used to separate discrete
            and continous numbers.
            (continuous case tend to has more types of uniq numbers.)

        Results
        --------------------
        cat_list : list of columns
            the list of categorical features.

        dis_num_list : list of columns
            the list of discrete numerical features.

        num_list : list of columns
            the list of continuous numerical features.

        '''
        cat_list = []
        dis_num_list = []
        num_list = []
        for i in data.columns.tolist():
            if data[i].dtype == 'object':
                cat_list.append(i)
            elif data[i].nunique() < uniq_disc_num_max:
                dis_num_list.append(i)
            elif i in special_list:     # if you want to add some special cases
                dis_num_list.append(i)
            else:
                num_list.append(i)
        return cat_list, dis_num_list, num_list
