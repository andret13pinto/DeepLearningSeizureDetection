import pyedflib
import nedc_print_labels.sys_tools.nedc_ann_tools as nat
import nedc_print_labels.sys_tools.nedc_file_tools
import sys
import os
import pandas as pd
import numpy as np


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
    :param fname: path of the .lbl or .tse file
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


def make_hdf_file(label, csv_file_information, channel):
    """
    Method for producting the hdf file
    from the edf files from the TUH-EEG
    dataset
    :param label: type of crisis
    :param csv_file_information: path to the csv
     file with the information regarding the signals
    :param channel:
    :return: produces a hdf file
    """
    data = []
    count = 0
    file_information = pd.read_csv(csv_file_information)
    for file in file_information.values[1:]:
        annot = load_annotations(file[channel], 0, 0)
        for index in range(len(annot)):  # for each part of signal
            if label in list(annot[index][2]):
                sf, s, lbls = nedc_load_edf(os.path.splitext(file[0])[0] + '.edf')
                init_sample = int(annot[index][0]*sf[channel])
                end_sample = int(annot[index][1]*sf[channel])
                signal = s[channel][init_sample:end_sample]
                signal = signal.tolist()
                signal.append(sf[channel])
                data.append(signal)
                print(len(data))
                if len(data) > 500:
                    count = count + 1
                    data = np.array(data)
                    pd.DataFrame(data).to_hdf(label + str(count) + "_training.hdf", key='stage', mode='w')
                    data = []
            else:
                continue
    data = np.array(data)
    if label is 'bckgd':
        pd.DataFrame(data).to_hdf(label + str(count) +"_training.hdf", key='stage', mode='w')
    else:
        pd.DataFrame(data).to_hdf(label + "_training.hdf", key='stage', mode='w')


def read_hdf_file(path):
    """
    Reads an hdf file from disk
    :param path: path to the file
    :return: numpy.array
    """
    data = pd.read_hdf(path)
    return data.values


def new_trainingData_csv(list_class, window, overlap, samples_per_class):
    """
    New method to produce the csv to feed into
    the model. It segments each sample according
    to the window and overlap chosen.
    The array procuded will have as many rows determined
    by the number of classes and samples per class
    :param list_class: list with all the classes
    wanted in the data array
    :param window: size of the window to split the signal
    into parts
    :param overlap: overlap between windows
    :param samples_per_class: number of samples for each class
    wanted in the data array
    :return: produces csv file
    """
    overlap = overlap*window
    signal_dic = {}  # temporary dic just to help counting the number of instances per class
    for c in list_class:
        hdf_path = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/"
        signal_dic[c] = []
        hdf_path = hdf_path + str(c) + "_training.hdf"
        hdf_data = read_hdf_file(hdf_path)
        for row in hdf_data:
            row = row[0] # necessary because of the dimensions
            fs = row[-1]
            init = 0
            end = int(window * fs)
            nr_samples = len(row)
            while end < nr_samples and len(signal_dic[c]) < samples_per_class:
                signal_dic[c].append(row[init:end])
                init = int(init + overlap*fs)
                end = int(end + overlap*fs)
    new_list = []
    for key in signal_dic.keys():
        for row in signal_dic[key]:
            row.append(key)
            new_list.append(row)
    data = np.array(new_list)
    print(data.shape)
    pd.DataFrame(np.array(new_list)).to_csv(str(len(list_class))+"class-" + str(samples_per_class)+"sam-"+str(window)+"w-"
                                            +str(overlap)+"ov.csv")


new_trainingData_csv(["tnsz","spsz","fnsz"], 2, 0.75, 50)
