import numpy as np
import pandas as pd
# from sklearn.preprocessing import Imputer

from tqdm import tqdm
from logging import getLogger
from collections import Counter

from data_processing.imputers import CustomImputer


class Preprocessor:
    r'''

    later I might implement this function as it holds all dataframes
    and process all data in its own eco system.

    '''

    def __init__(self, logger=None):
        self.logger = getLogger(logger.name).getChild(
            self.__class__.__name__) if logger else None

    def impute(self, target_df, target_column_list, missing_values="NaN",
               strategy='mean', axis=0, verbose=0, copy=True):
        r'''


        Parameters
        --------------------
        target_df : DataFrame
            the dataframe on which you want to fill the missing values.
        target_column_list : list of strings
            the columns for which you want to fill the missing values.
        missing_values : integer or “NaN”, optional (default=”NaN”)
            The placeholder for the missing values.
            All occurrences of missing_values will be imputed.
            For missing values encoded as np.nan, use the string value “NaN”.
        strategy : string, optional (default=”mean”)
            The imputation strategy.
            If “mean”, then replace missing values using
            the mean along the axis.
            If “median”, then replace missing values using
            the median along the axis.
            If “most_frequent”, then replace missing using
            the most frequent value along the axis.
        axis : integer, optional (default=0)
            The axis along which to impute.
            If axis=0, then impute along columns.
            If axis=1, then impute along rows.
        verbose : integer, optional (default=0)
            Controls the verbosity of the imputer.
        copy : boolean, optional (default=True)
            If True, a copy of X will be created.
            If False, imputation will be done in-place whenever possible.
            Note that, in the following cases, a new copy will always be made,
            even if copy=False:
                If X is not an array of floating values;
                If X is sparse and missing_values=0;
                If axis=0 and X is encoded as a CSR matrix;
                If axis=1 and X is encoded as a CSC matrix.

        Results
        --------------------
        res_df : DataFrame
            the dataframe which is filled missing value.
        '''
        if self.logger:
            self.logger.info('imputing missing data...')
            self.logger.debug('imputing target categories : {}'.format(
                target_column_list))
        # imputer = Imputer(missing_values="NaN",
        #                  strategy='mean', axis=0, verbose=0, copy=True)
        imputer = CustomImputer(strategy=strategy)
        target_df[target_column_list] = imputer.fit_transform(
            target_df[target_column_list])
        if self.logger:
            self.logger.info('data imputing done.')
        return target_df

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

    def foldout(self, train_df, test_size, seed=0):
        self.logger.info(
            'foldout training data (test size : {})'.format(test_size))
        np.random.seed(seed)

        df_size = train_df.shape[0]
        loc_index = np.arange(df_size)
        np.random.shuffle(loc_index)
        res_train_df = train_df.iloc[loc_index[:int(df_size * test_size)]]
        res_val_df = train_df.iloc[loc_index[int(df_size * test_size):]]

        return res_train_df, res_val_df

    def onehot_encoding(self, target_df,
                        additional_features=[], drop_first=True):
        r'''
        Encode the categorical columns of the input df into onehot style.
        This function recognizes the columns as categorical ones if the type
                is "object", so if you want to specify the other types of
                columns as categorical, please use additional_features.
        This can be implemented using categorical encoder of sklearn,
        but I use my own code for studying.


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
        original_columns = list(target_df.columns)
        for col in tqdm(target_df.columns.values):
            if target_df[col].dtype == 'object'\
                    or col in additional_features:
                if self.logger:
                    self.logger.debug('encoding the category {}'.format(col))
                tmp = pd.get_dummies(
                    target_df[col], col, drop_first=drop_first)
                for col2 in tmp.columns.values:
                    target_df[col2] = tmp[col2].values
                target_df.drop(col, axis=1, inplace=True)
        new_columns = [c for c in target_df.columns
                       if c not in original_columns]
        return target_df, new_columns

    def down_sampling(self, target_df, target_column):
        r'''
        down-sample the values.

        Parameters
        --------------------
        target_df : DataFrame
            the dataframe to which you want to apply onehot encoding.
        target_column : (basically) strings
            the names of columns based on which you want to apply
            down sampling

        Results
        --------------------
        target_df : DataFrame
            the dataframe which is applied down sampling
        '''
        if self.logger:
            self.logger.info('applying down sampling...'.format())
        counter = Counter(target_df[target_column].values)
        mincnt = min(counter.values())
        target_df = pd.concat([
            target_df[
                target_df[target_column] == sample_value].sample(mincnt)
            for sample_value in target_df[target_column].unique()
        ])

        return target_df
