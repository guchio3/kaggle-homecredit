import pandas as pd
import numpy as np

from sklearn.externals import joblib

from logging import getLogger
import datetime


class DataIO:
    r'''

    load data
    '''

    def __init__(self, logger=None):
        self.logger = getLogger(logger.name).getChild(
            self.__class__.__name__) if logger else None
        self.tdatetime = datetime.datetime.now()
        self.current_time = self.tdatetime.strftime('%Y-%m-%d_%H-%M-%S')

    def read_csv(self, filename):
        if self.logger:
            self.logger.info('read csv file from {}'.format(filename))
        df = pd.read_csv(filename)
        if self.logger:
            self.logger.info('successfully read')
        return df

    def read_csvs(self, filenames_dict):
        r'''
        read list of csvs.

        Parameters

        Return
        '''
        res_dict = {key: self.read_csv(filenames_dict[key])
                    for key in filenames_dict}
        return res_dict

    def save_csv(self, target_df, filename, index, withtime=False):
        if withtime:
            filename = filename.split('.')[0] + \
                       '_{}'.format(self.current_time) + filename.split('.')[1]
        if self.logger:
            self.logger.info('saving dataframe to {}'.format(filename))
        target_df.to_csv(filename, index=index)
        if self.logger:
            self.logger.info('successfully saved')

    def save_trained_params():
        r'''
        Save the trained parameters.
        In addition, this function saves meta-info of the params if required.

        '''
        with open('../trained_models/{}_{}_{}.info.txt'.format(clf.__class__.__name__, current_time, cnt), 'w') as param_file:
            param_file.write(json.dumps(params))
            joblib.dump(clf, '../trained_models/{}_{}_{}.pkl'.format(clf.__class__.__name__, current_time, cnt))
