from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

from logging import getLogger


class IntervalEvaluation(Callback):
    '''
    refered the below url.
    https://gist.github.com/smly/d29d079100f8d81b905e
    '''

    def __init__(self, validation_data=(), interval=1, logger=None):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        assert logger, 'ERROR: please set logger.'
        self.logger = getLogger(logger.name).getChild(
            self.__class__.__name__) if logger else None

#    def on_batch_end(self, batch, logs={}):
#        if(self.include_on_batch):
#            logs['roc_auc_val'] = float('-inf')
#            if(self.validation_data):
#                logs['roc_auc_val'] = \
#                    roc_auc_score(
#                    self.validation_data[1],
#                    self.model.predict(
#                        self.validation_data[0],
#                        batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            # y_pred = self.model.predict_proba(self.X_val, verbose=0)
            roc_auc_val = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = roc_auc_val
            self.logger.info(
                    "interval evaluation - epoch: {:d} - score: {:.6f}"
                    .format(epoch, roc_auc_val))
