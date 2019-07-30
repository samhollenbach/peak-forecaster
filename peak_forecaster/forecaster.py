import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow.keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def peak_loss(y_true, y_pred):
    y_dif = y_true - y_pred
    std = K.std(y_dif)
    mean = K.mean(y_dif)
    std_outlier = 2
    y_pos = tf.boolean_mask(y_dif, K.greater(y_dif, 0))
    y_neg = tf.boolean_mask(y_dif, K.less(y_dif, 0))
    y_non_outlier_pos = tf.boolean_mask(y_pos, K.less(y_pos, std_outlier*std))
    y_non_outlier_neg = tf.boolean_mask(y_neg, K.less(K.abs(y_neg), std_outlier*std))
    y_outlier_pos = tf.boolean_mask(y_pos, K.greater(y_pos, std_outlier*std))
    y_outlier_neg = tf.boolean_mask(y_neg, K.greater(K.abs(y_neg), std_outlier*std))

    # return K.mean(K.concatenate([10.0 * K.pow(y_non_outlier_pos, 2), K.pow(y_non_outlier_neg, 2), 10* K.pow(y_outlier_pos, 4),
    #                              K.pow(K.abs(y_outlier_neg), 4)]))
    return K.mean(K.concatenate([7.5 * K.pow(y_pos, 2), K.pow(y_neg, 2)]))
    # return K.mean(K.square(y_dif / std))
    # return K.var(K.square(y_dif))

def build_model(input_shape, output_shape):
  model = keras.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(input_shape,),
                 kernel_regularizer=regularizers.l2(0.03)),
    layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.3)),
    layers.Dense(output_shape)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

    # mean_squared_error
    # logcosh
  # mean_squared_logarithmic_error
  model.compile(loss=peak_loss,
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model



def train_model(model, train_x, train_y, test_x, test_y):
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

    # EPOCHS = 50000
    EPOCHS = 2000

    history = model.fit(
      train_x, train_y,
      epochs=EPOCHS, verbose=0,
        validation_data=(test_x, test_y),
      callbacks=[PrintDot()])
    print()

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    return history
