import preprocessing
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def one_hot_enconding(data):
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(), [-1])], remainder='passthrough')
    data = np.array(columnTransformer.fit_transform(data), dtype=np.str)
    return data


def load_data(csv_file, nr_classes):
    data = pd.read_csv(csv_file, header = None)
    data = data.values[1:, 1:] # strip off the first column and row
    data = one_hot_enconding(data)
    data = data.astype(np.float)
    X = data[:, nr_classes:]  # strip off indexes
    y = data[:, 0:nr_classes]
    X = normalize(X, axis=-1, order=2)  # L2 norm?
    return X, y


def CNN1(name_to_save):
    X,y = load_data('3classx1000sam.csv', 3)
    print(X.shape)
    X = np.reshape(X, (X.shape[0],X.shape[1],1))
    print(X.shape)
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(32, 3, input_shape=[X.shape[1], 1],  activation='relu'))
    model.add(layers.MaxPooling1D(pool_size = 2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation = 'relu'))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics =['accuracy'])
    model.fit(X, y, batch_size=20, epochs = 20, shuffle=True)
    model.save(name_to_save + str(time.time()))


def fullConnectedModel():
    X, y = load_data('3classx1000sam.csv', 3)
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, input_dim = X.shape[1], activation = 'relu'))
    model.add(layers.Dense(3, activation= 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=20)
    model.save("3classes-5000samples-{}".format(str(time.time())))
    return model


#CNN1("2layerCNN-3class-5000sam")
#fullConnectedModel()
