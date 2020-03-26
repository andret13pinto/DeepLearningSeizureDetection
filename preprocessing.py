import pyedflib
import nedc_print_labels.sys_tools.nedc_ann_tools as nat
import sys
import os
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import LabelEncoder
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt


def nedc_load_edf(fname_a):
    """
    :param fname_a: edf file load signal
    :return: list with sf for each montage,
    signal in each montage,
    montage names of montages
    """

    # open an EDF file
    #
    try:
        fp = pyedflib.EdfReader(fname_a)
    except IOError:
        print ("%s (%s: %s): failed to open %s" % \
            (sys.argv[0], __name__, "nedc_load_edf", fname_a))
        exit(-1)

    # get the metadata that we need:
    #  convert the labels to ascii and remove whitespace
    #  to make matching easier
    #
    num_chans = fp.signals_in_file
    labels_tmp = fp.getSignalLabels()
    labels = [str(lbl.replace(' ', '')) for lbl in labels_tmp]

    # load each channel
    #
    sig = []
    fsamp = []
    for i in range(num_chans):
        sig.append(fp.readSignal(i))
        fsamp.append(fp.getSampleFrequency(i))

    # exit gracefully
    #
    return (fsamp, sig, labels)


def load_annotations(fpath, lev, sub):
    """
    :param fpath: path of the .lbl or .tse file
    :param lev: level in the montage - 0
    :param sub: sublevel in the montage - 0
    :return: list with the intervals of each period and a
    OrdDic with the type of crisis and probability (always 1).
    """
    ann = nat.Annotations()
    status = ann.load(fpath)
    num_files_proc = 0
    if not status:
        print ("%s (%s: %s): error loading label file (%s)" % \
            (sys.argv[0], __name__, "main", fpath))
    if status:
        status = ann.display(lev, sub)
        if status:
            num_files_proc += 1
    # get the annotations
    #
    channel = -1
    annotations = ann.get(lev, sub, channel)
    return annotations


def get_signal_information(folder_path):
    """

    :param folder_path: folder path to the annotations
    :return: list of lists of annotations, each row is
    a new list with the filepath and the number of periods
    in the signal.
    Ex : ["filepath", 3]
    """
    ann_list = [["File_name", "Nr_periods", "Fs"]]
    for subfolder in os.listdir(folder_path):
        subfolder_path = folder_path + str(subfolder)
        for number in os.listdir(subfolder_path):
            number_path = subfolder_path + "/" + str(number)
            for second_number in os.listdir(number_path):
                second_number_path = number_path + "/" + str(second_number)
                for session in os.listdir(second_number_path):
                    session_path = second_number_path + "/" + str(session)
                    for file in os.listdir(session_path):
                        file_path = session_path + "/" + str(file)
                        if file_path.endswith('.tse'):
                            print(file_path)
                            annot = load_annotations(file_path, 0, 0)
                            fs, s, lbs = nedc_load_edf(os.path.splitext(file_path)[0] + '.edf')
                            ann_list.append([file_path, len(annot), fs[0]])
    return ann_list


def make_excel_information(filename, folder_path):
    """
    :param filename: name of the output xlsx file
    ex: 'my_excel_file.xlsx' (needs the extension)
    :param folder_path: folder path to the annotations
    :return: creates excel file in current directory
    with the name of the file, the number of periods
    and the sampling frequency of the selected channel
    """
    ann_list = get_signal_information(folder_path)
    df = pd.DataFrame(ann_list)
    print(df)
    return df.to_excel(filename, index=False)


def make_h5_file(label, csv_file_information, electrode):
    """
    Method for producting the hdf file
    from the edf files from the TUH-EEG
    dataset

    :param label: type of crisis
    :param csv_file_information: path to the csv
     file with the information regarding the signals
    :param electrode: electrode chosen
    :return: produces a hdf file
    """
    possible_montages = ["EEG" + electrode + "-LE", "EEG" + electrode + "-AR", "EEG" + electrode + "-REF"]
    data = []
    file_info = pd.read_csv(csv_file_information)
    for f in file_info.values[1:]:
        ann = load_annotations(f[0], 0, 0)
        for index in range(len(ann)):  # for each part of signal
            if label == list(ann[index][2])[0]:
                sf, s, lbls = nedc_load_edf(os.path.splitext(f[0])[0] + '.edf')
                channel = [idx for idx in range(len(lbls)) if lbls[idx] in possible_montages]
                montage = lbls[channel[0]]
                init_sample = int(ann[index][0]*sf[channel[0]])
                end_sample = int(ann[index][1]*sf[channel[0]])
                sign = s[channel[0]][init_sample:end_sample]
                sign = np.append(sign, sf[channel[0]])
                data.append(sign)
                print(len(data))
            else:
                continue
    hf = h5py.File('{0}-{1}.h5'.format(label,electrode), 'w')
    dt = h5py.vlen_dtype(np.dtype('float64'))
    hf.create_dataset(label, data=data, dtype=dt)
    hf.close()


def read_h5_file(path, label):
    """
    Reads an hdf file from disk

    :param path: path to the file
    :return: numpy.array
    """
    file = h5py.File(path,'r')
    if label == "bckg":  # account for error naming h5 dataset
        label = "bckgd"
    data = file.get(label)
    return np.array(data)


def make_randomized_window_samples(number_samples, Tdata, desired_fs, window):
    """
    Method to produce infinite samples randomly from the
    set of intervals provided

    :param number_samples: number of samples wanted
    :param Tdata: list of complete signals
    :param desired_fs: sampling frequency wanted
    :param window: window size for each sample
    :return: a list with all samples each with (window) window sized
    """
    samples_list = []
    while len(samples_list) < number_samples or number_samples is None:
        print(len(samples_list))
        random_index = np.random.randint(0, len(Tdata) - 1)  # each time produce different data
        row = Tdata[random_index]
        fs = row[-1]
        if fs != desired_fs:  # resampling the signal into the smallest sample size
            ratio = int(fs / desired_fs)
            signal.resample_poly(row, 10, 10 * ratio)
            fs = desired_fs
        if len(row) >= window * fs:
            random_start = np.random.randint(0, len(row) - window * fs)
        else:
            continue
        samples_list.append(row[random_start:random_start + int(window * fs)])
    return samples_list


def make_limited_window_samples(number_samples, Tdata, desired_fs, window, overlap):
    """
    Method to produce limited samples from the
    set of intervals provided

    :param number_samples: number of samples wanted
    :param Tdata: list of complete signals
    :param desired_fs: sampling frequency wanted
    :param window: window size for each sample
    :param overlap:
    :return: a list with all samples each with (window) window sized
    """
    index = 0
    samples_list = []
    break_mode = False
    for row in Tdata:
        if break_mode:
            break
        fs = list(row).pop(-1)
        row = row[:-1]
        if fs != desired_fs:  # resampling the signal into the smallest sample size
            ratio = int(fs / desired_fs)
            signal.resample_poly(row, 10, 10 * ratio)
            fs = desired_fs
        init = 0
        end = int(window * fs)
        while end < len(row):
            if number_samples is not None:
                if len(samples_list) == number_samples:
                    break_mode = True
                    break
                else:
                    samples_list.append(row[init:end])
                    init = int(init + overlap)
                    end = int(end + overlap)
            else:
                samples_list.append(row[init:end])
                init = int(init + overlap)
                end = int(end + overlap)
    return samples_list


def fromDictoArray(dic):
    """
    Convert from dictionary with samples,
    each key with a different class
    to array with all samples
    :param dic:  dictionary with samples,
    each key with a different class
    :return: array with all samples
    """
    temp_list = []
    for key in dic:
        ind = list(dic.keys()).index(key)
        for row in dic[key]:
            row = np.append(row, ind)
            temp_list.append(row)
    return np.array(temp_list)


def newData_csv(list_class, window, electrode, samples_per_class = None, desired_fs=250, validation_split=None):
    """
    Produces a csv file to provide training data to the algorithm,
    this data is produced by the make_randomized_window_samples
    which produces unlimited samples

    :param list_class: list with classes of EEG signal to input into the algorithm
    :param window: window of signal to classify
    :param samples_per_class: number of examples per class
    :param electrode: electrode-name from which the signals have
    been recorded
    :param desired_fs: desired sampling frequency of all files
    :param validation_split: percentage of the data that goes to the validation set, can be None
    :return: a csv file
    """
    base_path = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/FP2/"
    name_tosave_train = "{0}-{1}samp-{2}w-{3}-fs-Training.csv".format(list_class,samples_per_class, window, desired_fs)
    name_tosave_valid = "{0}-{1}samp-{2}w-{3}-fs-Validation.csv".format(list_class, samples_per_class, window,
                                                                            desired_fs)
    signal_dic_train = {}  # temporary dic just to help counting the number of instances per class
    if validation_split is not None:
        signal_dic_validation = {}
    for c in list_class:
        print(c)
        signal_dic_train[c] = []
        if validation_split is not None:
            signal_dic_validation[c] = []
        class_path = base_path + c + "-" + electrode + ".h5"
        hdf_data = read_h5_file(class_path, c)
        if validation_split is not None:
            mask = np.random.choice([False, True], len(hdf_data), p=[1-validation_split, validation_split], replace=True)
            print(mask)
            validation_data = hdf_data[mask]
            training_data = hdf_data[~mask]
        else:
            training_data = hdf_data
        signal_dic_train[c] = make_randomized_window_samples(samples_per_class, training_data, desired_fs, window)
        print([len(signal_dic_train[k]) for k in signal_dic_train.keys()])
        if validation_split is not None:
            signal_dic_validation[c] = make_randomized_window_samples(int(samples_per_class*validation_split), validation_data, desired_fs, window)
            print([len(signal_dic_validation[k]) for k in signal_dic_validation.keys()])
    final_Training_data = fromDictoArray(signal_dic_train)
    final_Validation_data = fromDictoArray(signal_dic_validation)
    print(final_Training_data[:,-1])
    print(final_Validation_data[:, -1])
    np.savetxt(name_tosave_train, final_Training_data, delimiter=',')
    np.savetxt(name_tosave_valid, final_Validation_data, delimiter=',')


def limitedNewData_csv(list_class, window, overlap, electrode, samples_per_class = None, desired_fs=250,validation_split = None):
    """
    Produces a csv file to provide training data to the algorithm,
    this training data is produced by the method
    make_limited_window_samples which produces as many samples
    per class as possible if a higher number
    according to the number of samples_per class wanted

    :param list_class: list with classes of EEG signal to input into the algorithm
    :param window: window of signal to classify
    :param overlap: percentage of the window which is overlapped to the next sample
    :param samples_per_class: number of examples per class
    :param electrode:  electrode-name from which the signals have
    been recorded
    :param desired_fs: desired sampling frequency of all files
    :param validation_split: percentage of the data that goes to the validation set, can be None
    :return: csv file with data
    """
    base_path = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/FP2/"
    name_tosave_train = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/csv_files/" \
                        "FP2/{0}-{1}samp-{2}w-{3}-fs-TrainingLimited.csv".format(list_class,
                                                                                 samples_per_class, window, desired_fs)
    name_tosave_valid = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/FP2/csv_files/" \
                        "{0}-{1}samp-{2}w-{3}-fs-ValidationLimited.csv".format(list_class, samples_per_class, window,
                                                                        desired_fs)
    overlap = int(overlap*window)
    signal_dic_train = {}  # temporary dic just to help counting the number of instances per class
    if validation_split is not None:
        signal_dic_validation = {}
    for c in list_class:
        print(c)
        signal_dic_train[c] = []
        if validation_split is not None:
            signal_dic_validation[c] = []
        class_path = base_path + c + "-" + electrode + ".h5"
        hdf_data = read_h5_file(class_path, c)
        if validation_split is not None:
            mask = np.random.choice([False, True], len(hdf_data), p=[1 - validation_split, validation_split],
                                    replace=True)
            print(mask)
            validation_data = hdf_data[mask]
            training_data = hdf_data[~mask]
        else:
            training_data = hdf_data
        signal_dic_train[c] = make_limited_window_samples(samples_per_class, training_data, desired_fs, window, overlap)
        print([len(signal_dic_train[k]) for k in signal_dic_train.keys()])
        if validation_split is not None:
            if samples_per_class is None:
                signal_dic_validation[c] = make_limited_window_samples(None,
                                                                       validation_data, desired_fs, window, overlap)
            else:
                signal_dic_validation[c] = make_limited_window_samples(int(samples_per_class * validation_split),
                                                                       validation_data, desired_fs, window, overlap)
            print([len(signal_dic_validation[k]) for k in signal_dic_validation.keys()])
    final_Training_data = fromDictoArray(signal_dic_train)
    final_Validation_data = fromDictoArray(signal_dic_validation)
    np.savetxt(name_tosave_train, final_Training_data, delimiter=',')
    np.savetxt(name_tosave_valid, final_Validation_data, delimiter=',')
