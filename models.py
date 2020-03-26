import preprocessing
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.utils import normalize
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
import logging
from sklearn.compose import ColumnTransformer


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
def load_data(csv_file, nr_classes):
    print("Loading data ...")
    data = np.loadtxt(csv_file, delimiter=',')
    print("Data loaded...")
    last_column = data[:, -1]
    last_column = last_column.reshape(len(last_column), 1)
    last_column = OneHotEncoder(sparse=False).fit_transform(last_column)
    data[:, -1] = last_column[:, 0]
    for i in range(nr_classes-1):
        data = np.append(data, last_column[:, i+1].reshape(len(last_column[:, 1]), 1), axis=1)
    X = data[:, :-nr_classes]  # strip off indexes
    y = data[:, -nr_classes:]
    print(y)
    #X = normalize(X, axis=0, order=1)  # L2 norm?
    return X, y


def visualize_data(csv_file, number_classes,number_samples):
    X,y = load_data(csv_file, number_classes)
    l = np.linspace(1000, number_classes*number_samples, 30, dtype=int)
    print(l)
    for index in l:
        plt.figure()
        plt.plot(X[index-1])
        plt.title(str(y[index-1,0:number_classes]))
        plt.show()


def CNN(training_data_path, list_classes, number_layers , window_size,
         filter_number_conv = 32, dense_layer_size = None , epochs=20, pooling_size = 2 , validation_data_path = None, balanced = True):
    clear_session()
    set_tf_loglevel(logging.FATAL)
    model_name = "{0}classes-CNN-{1}x{2}filt-{3}Dsize-{4}Poolsize-{5}W".format(list_classes,number_layers,
                                            filter_number_conv, dense_layer_size, pooling_size, window_size)
    while os.path.isdir(model_name):
        try:
            number = int(model_name.strip(".")[1])
            number += 1
            model_name = model_name.strip(".")[0] + "." +str(number)
        except:
            model_name = model_name + ".2"
    tensorboard = TensorBoard(log_dir ='logs/{}'.format(model_name))
    X_train,y_train = load_data(training_data_path, len(list_classes))
    print(X_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    if validation_data_path is not None:
        X_validation, y_validation = load_data(validation_data_path, len(list_classes))
        X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
    model = tf.keras.Sequential()
    for i in range(number_layers):
        model.add(layers.Conv1D(filter_number_conv, 3, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    if dense_layer_size is not None:
        model.add(layers.Dense(dense_layer_size, activation = 'relu'))
    model.add(layers.Dense(len(list_classes), activation='softmax'))
    weights = []
    for index in range(len(y_train[0])):
        count = len([y for y in y_train[:, index] if y == 1])
        weights.append(count/len(y_train[:, index]))
    weights =np.array(weights)
    print(weights)
    model.compile(loss=weighted_categorical_crossentropy(weights), optimizer="adam",
                  metrics=['accuracy', 'Recall', 'Precision'])
    if validation_data_path is not None:
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, shuffle=True, validation_data=(X_validation,y_validation),
                  callbacks=[tensorboard])
    else:
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, shuffle=True, validation_split=0.1,
                  callbacks=[tensorboard])
    model.summary()
    model.save(model_name)
    return model


def fullConnectedModel(number_classes):
    model_name = "FullConnectedModel-3classes-12000samples-{}".format(str(time.time()))
    X_train, y_train = load_data('training3-class-12000sam-2w-1.5ov.csv', number_classes)
    print(X_train)
    print(y_train)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, input_dim = X_train.shape[1], activation = 'relu'))
    model.add(layers.Dense(3, activation= 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','Recall','Precision'])
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=0.1,
              callbacks=[tensorboard], shuffle=True)
    model.save(model_name)
    return model


base_path = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/FP2/csv_files/"
CNN(base_path + "['absz', 'bckg']-20000samp-2w-250-fs-TrainingLimited.csv",
    ["absz","bckg"], 4, 2, filter_number_conv=64, pooling_size=3, epochs=10, validation_data_path= base_path + "['absz', 'bckg']-20000samp-2w-250-fs-ValidationLimited.csv", balanced=False)