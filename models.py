import segmenting
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

training_data = {"bckg":[]}
NR_FILES_PER_CLASS = 100
NR_CLASSES = 5
csv_path = "file_information.csv"
channel =  0
overlap = 1
window_size = 10
nr_intervals = 100,
nr_clases = 5


def produce_csv2(nr_intervals, nr_classes, csv_path, channel, overlap, window_size):
    """
    The method aims to create a csv file which will support the training of a model
    with a limited amount of data, bearing in mind that each class should have the same
    amount of samples.

    :param nr_intervals: number of intervals present in the csv file for each class
    :param nr_classes: number of different types of classification in the file
    :param csv_path: path to the original csv file with information to the characterisctics of
            each file in the dataset
    :param channel: channel or montage used, normally 0
    :param overlap: overlap between windows/interval x intervalwindow (0.75 x window)
    :param window_size: the size of each interval/window
    :return:
    """
    multiple_break = False
    file_information = pd.read_csv(csv_path)
    lengths = [len(v) for v in training_data.values()]
    for file in file_information.values[1:]:  # values returns a numpy representation of the Dataframe
        if file[2] != "256":
            # want only the files with 256 Hz
            continue
        if multiple_break:
            break
        if len(training_data["bckg"]) == nr_intervals and file[1] == "1":  # check if the next file has crisis
            continue
        else:
            ann = segmenting.get_annotations(file[0], 0, 0)
            fs, s, lbls = segmenting.nedc_load_edf(os.path.splitext(file[0])[0] + '.edf')
            file_data = segmenting.get_training_data_fromFile(s[channel], fs[channel], ann, overlap, window_size)
            for row in file_data:
                if row[-1] not in training_data.keys():
                    training_data[row[-1]] = [row[:-1]]
                else:
                    if len(training_data[row[-1]]) < nr_intervals:  # the number of lists in the list of lists, which is the number of rows
                        training_data[row[-1]].append(row[:-1])
                lengths = [len(v) for v in training_data.values()]
                count = len([number for number in lengths if number == nr_intervals])
                if count == nr_classes:
                    multiple_break = True
                    for key in list(training_data.keys()):
                        if len(training_data[key]) < nr_intervals:
                            del training_data[key]
                    break
            print(lengths)
    new_list = []
    for key in training_data.keys():
        for row in training_data[key]:
            row.append(key)
            new_list.append(row)
    data = np.array(new_list)
    print(data.shape)
    pd.DataFrame(np.array(new_list)).to_csv("3classx1000sam.csv")


def one_hot_enconding(data):
    columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(), [-1])], remainder='passthrough')
    data = np.array(columnTransformer.fit_transform(data), dtype=np.str)
    return data


def CNN1():
    data = pd.read_csv('3class.csv', header=None)
    X = data.values[1:,1:-1]  # strip off indexes
    y = data.values[1:,-1]
    X = normalize(X,axis = -1, order = 2)  # L2 norm?
    X = np.expand_dims(X, axis=2) # reshape (30, 2560) to (30, 2560, 1)
    print(X.shape)
    print(y)
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(10, 3, input_shape= [X.shape[1],1], activation='relu'))
    model.summary()
    model.add(layers.MaxPooling1D(pool_size = 2))
    model.summary()
    model.add(layers.Dense(30))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics =['accuracy'])
    model.fit(X,y,batch_size=20)


def fullConnectedModel():
    data = pd.read_csv('3classx1000sam.csv', header=None)
    data = data.values[1:,1:]
    data = one_hot_enconding(data) # one hot encoding at the beggining of array
    data = data.astype(np.float)  # convert from string to float
    X = data[:, 3:]  # strip off indexes
    y = data[:, 0:3]
    X = normalize(X, axis=-1, order=2)  # L2 norm?
    #X = np.expand_dims(X, axis=2)  # reshape (30, 2560) to (30, 2560, 1)
    print(X.shape)
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, input_dim = X.shape[1], activation = 'relu'))
    model.add(layers.Dense(3, activation= 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=20)
    model.save("3classes-5000samples-{}".format(str(time.time())))
    return model

