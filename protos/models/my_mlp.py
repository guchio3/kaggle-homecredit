import os
import numpy as np
import random as rn
import tensorflow as tf

# from keras import Model
from keras import regularizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping

from .callbacks.interval_evalation import IntervalEvaluation


class myMLPClassifier():
    def __init__(self, hidden_layer_sizes=(100, ),
                 batch_norm=(False, False), dropout=(0.0, 0.0, ),
                 activation='relu', solver='adam',
                 batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, alpha=0.0001,
                 power_t=0.5, max_iter=200,
                 verbose=1, random_state=None, tol=0.0001,
                 validation_fraction=0.1, eval_set=None):
        if random_state:
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(random_state)
            rn.seed(random_state)
            tf.set_random_seed(random_state)
        assert len(hidden_layer_sizes) + 1 >= len(dropout), \
            'invalid dropout setting : {}'.format(dropout)
        assert len(hidden_layer_sizes) + 1 >= len(batch_norm), \
            'invalid batch_norm setting : {}'.format(batch_norm)

#        self.hidden_layers = [Dense(l_size, activation=activation)
#                              for l_size in hidden_layer_sizes]
        self.hidden_layers = hidden_layer_sizes
        if len(hidden_layer_sizes) + 1 > len(dropout):
            dropout += tuple([dropout[-1] for _ in
                              range(len(hidden_layer_sizes) -
                                    len(dropout) + 1)])
#        self.dropout = [Dropout(d) for d in dropout]
        self.dropout = dropout
        if len(hidden_layer_sizes) + 1 > len(batch_norm):
            batch_norm += tuple([batch_norm[-1] for _ in
                                 range(len(hidden_layer_sizes) -
                                       len(batch_norm) + 1)])
#        self.batch_norm = [BatchNormalization(axis=-1) if b else None
#                           for b in batch_norm]
        self.batch_norm = batch_norm
        self.batch_size = 200 if batch_size == 'auto' else batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.power_t = power_t
        self.epochs = max_iter
        self.verbose = verbose
        self.tol = tol
        self.validation_split = validation_fraction
        self.validation_data = eval_set
        if solver == 'sgd':
            self.optimizer = SGD(lr=learning_rate_init, )
        elif solver == 'adam':
            self.optimizer = Adam(lr=learning_rate_init, )
        elif solver == 'rmsprop':
            self.optimizer = RMSprop(lr=learning_rate_init, )
        else:
            assert(NotImplementedError)

    def build(self, input_shape, output_shape):
        inputs = Input(shape=input_shape, )
        x = Dropout(self.dropout[0])(inputs)
        x = BatchNormalization(axis=-1)(x)
#        x = self.dropout[0](inputs)
#        x = self.batch_norm[0](x) if self.batch_norm[0] else x
        for l, d, b in zip(
                self.hidden_layers, self.dropout[1:], self.batch_norm[1:]):
            x = Dense(l, activation='relu')(x)
            x = Dropout(d)(x)
            x = BatchNormalization(axis=-1, )(x) if b else x
#            x = l(x)
#            x = d(x)
#            x = b(x) if b else x
#        if output_shape == (1, ):
        if True:
            x = Dense(1, activation='sigmoid')(x)
        else:
            x = Dense(output_shape, activation='softmax')(x)
        self.model = Model(inputs, x)
        self.model.compile(self.optimizer, loss='binary_crossentropy',
                           metrics=['accuracy'],)
#                           kernel_regularizer=regularizers.l2(self.alpha))

    def fit(self, x, y, eval_set=None):
        self.build(x[0].shape, y[0].shape)
        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001, patience=2))
        if eval_set:
            callbacks.append(IntervalEvaluation(validation_data=eval_set))
        self.model.fit(x=x, y=y,
                       validation_data=eval_set,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.validation_split,
                       verbose=self.verbose,
                       callbacks=callbacks,
                       )

    def predict_proba(self, x):
        pred = self.model.predict(x)
        return np.concatenate([1. - pred, pred], axis=1)
