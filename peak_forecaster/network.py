import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow.keras.backend as K
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Network(object):

    def __init__(self, params):

        default_params = {
            'layers': 2,
            'activation': keras.layers.LeakyReLU(alpha=0.2),
            'nodes': [32, 32],
            'optimizer': tf.keras.optimizers.Adamax,
            'learning_rate': 0.001,
            'loss_positive_scalar': 7.5,
        }

        self.TRAIN_EPOCHS = 2000
        self.params = {**default_params, **params}
        self.model = None


    def build_layers(self, input_shape, output_shape):
        network_layers = []
        for i in range(self.params['layers']):
            n = self.params['nodes'][i]
            act = self.params['activation']
            if i == 0:
                l = layers.Dense(n, activation=act,
                                 input_shape=(input_shape,),
                                 kernel_regularizer=regularizers.l2(0.03))
            else:
                l = layers.Dense(n, activation=act,
                                 kernel_regularizer=regularizers.l2(0.03))
            network_layers.append(l)
        network_layers.append(layers.Dense(output_shape))
        return network_layers

    def peak_loss(self, y_true, y_pred):
        y_dif = y_true - y_pred
        std = K.std(y_dif)
        mean = K.mean(y_dif)
        std_outlier = 2
        y_pos = tf.boolean_mask(y_dif, K.greater(y_dif, 0))
        y_neg = tf.boolean_mask(y_dif, K.less(y_dif, 0))
        y_non_outlier_pos = tf.boolean_mask(y_pos, K.less(y_pos,
                                                          std_outlier *
                                                          std))
        y_non_outlier_neg = tf.boolean_mask(y_neg, K.less(K.abs(y_neg),
                                                          std_outlier *
                                                          std))
        y_outlier_pos = tf.boolean_mask(y_pos, K.greater(y_pos,
                                                         std_outlier *
                                                         std))
        y_outlier_neg = tf.boolean_mask(y_neg, K.greater(K.abs(y_neg),
                                                         std_outlier *
                                                         std))

        # return K.mean(K.concatenate([10.0 * K.pow(y_non_outlier_pos,
        # 2), K.pow(y_non_outlier_neg, 2), 10* K.pow(y_outlier_pos, 4),
        #                              K.pow(K.abs(y_outlier_neg), 4)]))

        y_pos_true = tf.boolean_mask(y_true, K.greater(y_dif, 0))
        y_pos_scale = K.square((y_pos_true / K.mean(y_pos_true)))
        y_neg_true = tf.boolean_mask(y_true, K.less(y_dif, 0))
        # y_neg_scale = K.square((y_neg_true / K.mean(y_neg_true)))
        y_neg_scale = 1.0
        return K.mean(K.concatenate([self.params['loss_positive_scalar'] * y_pos_scale * K.pow(y_pos, 2),
                                     y_neg_scale * K.pow(y_neg, 2)]))
        # return K.mean(K.concatenate([7.5 * K.pow(y_pos, 2), K.pow(y_neg, 2)]))
        # return K.mean(K.square(y_dif / std))
        # return K.var(K.square(y_dif))

    def build_model(self, input_shape, output_shape):

        self.model = keras.Sequential(self.build_layers(input_shape, output_shape))
        # model = keras.Sequential([
        #     layers.Dense(16, activation='elu', input_shape=(input_shape,),
        #                  kernel_regularizer=regularizers.l2(0.03)),
        #     layers.Dense(16, activation='elu',
        #                  kernel_regularizer=regularizers.l2(0.03)),
        #     layers.Dense(output_shape)
        # ])

        # optimizer = tf.keras.optimizers.Adamax(0.001)
        optimizer = self.params['optimizer'](self.params['learning_rate'])

        # mean_squared_error
        # logcosh
        # mean_squared_logarithmic_error
        self.model.compile(loss=self.peak_loss,
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return self.model

    def train_model(self, train_x, train_y):
        # Display training progress by printing a single dot for each
        # completed epoch
        class ProgressBar(keras.callbacks.Callback):
            def __init__(self, total):
                self.total = total
                self.bar_length = 64
                self.current_epoch = 1

            def create_bar(self, cur):
                frac = cur / self.total
                done_num = math.ceil(self.bar_length * frac)
                done = '█' * done_num
                not_done = '_' * (self.bar_length - done_num)
                print(f'\r|{done}{not_done}| {100 * frac:.2f}%',
                      end='')

            def on_epoch_end(self, epoch, logs):
                self.current_epoch = epoch + 1
                if epoch % 10 == 0:
                    self.create_bar(epoch)


            def on_train_end(self, logs=None):
                self.create_bar(self.current_epoch)
                if self.current_epoch < self.total:
                    print(f"\nNOTICE: Training stopped early"
                          f" at epoch {self.current_epoch} / {self.total}")
                print("\n\n")

        # The patience parameter is the amount of epochs to check for
        # improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=500)

        print("\n##### Training Progress #####\n")
        history = self.model.fit(
            train_x, train_y,
            epochs=self.TRAIN_EPOCHS, verbose=0,
            validation_split=0.2,
            callbacks=[early_stop, ProgressBar(self.TRAIN_EPOCHS)])


        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        return history



