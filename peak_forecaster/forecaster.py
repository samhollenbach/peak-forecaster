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
    y_pos = tf.boolean_mask(y_dif, K.greater(y_dif, 0))
    y_neg = tf.boolean_mask(y_dif, K.less(y_dif, 0))

    return K.mean(K.concatenate([K.pow(y_pos, 2), y_neg]))


def build_model(input_shape, output_shape):
  model = keras.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(input_shape,),
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu'),
    layers.Dense(output_shape)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

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
        if epoch % 100 == 0: print('')
        print('.', end='')

    EPOCHS = 1000

    history = model.fit(
      train_x, train_y,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
        validation_data=(test_x, test_y),
      callbacks=[PrintDot()])
    print()

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    return history


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error kW')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,10])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$kW^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,200])
  plt.legend()
  plt.show()