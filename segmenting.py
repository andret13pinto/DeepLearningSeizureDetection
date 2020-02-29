import pyedflib
import nedc_print_labels.sys_tools.nedc_ann_tools as nat
import sys
import os
import numpy as np
import pickle
import pandas as pd
import glob
from collections import OrderedDict


# example of a  class_dic
# labels_dic = {background : [0,120], seizure_a :[120,1000]}
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


def get_annotations(fpath, lev, sub):
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


def get_annotation_list(folder_path):
    """

    :param folder_path: folder path to the annotations
    :return: list of lists of annotations, each row is
    a new list with the filepath and the number of periods
    in the signal.
    Ex : ["filepath", 3]
    """
    ann_list = [["File_name", "Nr_periods"]]
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
                            annot = get_annotations(file_path, 0, 0)
                            ann_list.append([file_path,len(annot)])
    return ann_list


def get_excel_information(filename, folder_path):
    """
    :param filename: name of the output xlsx file
    ex: 'my_excel_file.xlsx'
    :param folder_path: folder path to the annotations
    :return: creates excel file in current directory
    """
    ann_list = get_annotation_list(folder_path)
    df = pd.DataFrame(ann_list)
    print(df)
    return df.to_excel(filename, index=False)


def get_training_data_fromFile(signal, sf, labels_list, overlap, window_size):
    """
    Creates training data for a single edf file

    sig: complete signal,

    sf: sample frequency (hz),

    labels_dic : an ordered dictionary which has all the segmentation of the signals,
    as well as the start and end of each part,
    example {background : [0,120], seizure_a :[120,1000]},

    overlap: signal overlap in seconds,

    window_size: window_size in seconds.

    """
    data = []
    nr_samples = len(signal)
    init = 0  # in samples
    end = window_size*sf  # in samples
    label_index = 0  # the position of the label in the dic
    current_label = list(labels_list[label_index][2])[0]
    while end < nr_samples:
        if init/sf < labels_list[label_index][1] and end/sf < labels_list[label_index][1]:
            temp_list = list(signal[init:end]).copy()
            temp_list.append(current_label)
            data.append(temp_list)
        else:
            label_index += 1
            init = int(labels_list[label_index][0]*sf)
            end = init + window_size * sf
            current_label = list(labels_list[label_index][2])[0]
            continue
        init = init + overlap*sf
        end = end + overlap*sf
    return data


def get_all_training_data(folder_path, channel, overlap, window_size):
    number_file = 0
    training_data = []
    for number in os.listdir(folder_path):
        number_path = folder_path + "/" + str(number)
        for second_number in os.listdir(number_path):
            second_number_path = number_path + "/" + str(second_number)
            for session in os.listdir(second_number_path):
                session_path = second_number_path + "/" + str(session)
                os.chdir(session_path)
                print(session_path)
                for file in glob.glob("*.edf"):  # remove the extension of the first file in the list
                    number_file +=1
                    print(str(number_file) + "of 4597:"  + str(int(number_file/4597)))
                    file_path = session_path + "/" + file
                    fsamp, sig, labels = nedc_load_edf(file_path)
                    tse_path = os.path.splitext(file_path)[0]
                    annotations = get_annotations(tse_path + ".tse", 0, 0)
                    training_data.extend(get_training_data_fromFile(sig[channel], fsamp[channel], annotations, overlap, window_size))
    return pd.DataFrame(training_data)


t_data = get_all_training_data("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar", 0 , 1, 10)
with open("test.txt", "wb") as fp:   #Pickling
    pickle.dump(t_data, fp)

# ann = get_annotations("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/134/00013407/s001_2015_09_28/00013407_s001_t004.tse",0,0)
# print(len(ann))
# print(ann[0][2])

# print(get_annotations("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/134/00013407/s001_2015_09_28/00013407_s001_t004.tse",0,0)[0])


fsamp, sig, labels = nedc_load_edf("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/134/00013407/s001_2015_09_28/00013407_s001_t004.edf")

#annotations = get_annotations("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/134/00013407/s001_2015_09_28/00013407_s001_t004.tse",0,0)
#data = get_training_data_fromFile(sig[0],fsamp[0], annotations, 1, 2)
#print(np.array(data).shape)



