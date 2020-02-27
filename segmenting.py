import pyedflib
import nedc_print_labels.sys_tools.nedc_ann_tools as nat
import sys
import os
from collections import OrderedDict


# example of a  class_dic
# labels_dic = {background : [0,120], seizure_a :[120,1000]}

def get_annotations(fname, lev, sub):
    ann = nat.Annotations()
    status = ann.load(fname)
    num_files_proc = 0
    print(status)
    if not status:
        print ("%s (%s: %s): error loading label file (%s)" % \
            (sys.argv[0], __name__, "main", fname))
    if status:
        status = ann.display(lev, sub)
        if status:
            num_files_proc += 1
    # get the annotations
    #
    channel = -1
    annotations = ann.get(lev, sub, channel)
    return annotations


def nedc_load_edf(fname_a):

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


def get_training_data(sig, sf, labels_dic, overlap, window_size):
    """
    sig: complete signal,

    sf: sample frequency (hz),

    labels_dic : an ordered dictionary which has all the segmentation of the signals,
    as well as the start and end of each part,
    example {background : [0,120], seizure_a :[120,1000]},

    overlap: signal overlap in seconds,

    window_size: window_size in seconds.

    """
    data = []
    nr_samples = len(sig)
    init = 0  # in samples
    end = window_size*sf  # in samples
    label_index = 0  # the position of the label in the dic
    current_label = labels_dic.keys()[label_index]
    while end < nr_samples:
        if init/sf < current_label.values()[-1] and end/sf < current_label.values()[-1]:
            temp_list = sig[init:end].append(current_label)
            data.append(temp_list)
        else:
            init = end
            end = end + window_size * sf
            label_index += 1
            current_label = labels_dic.keys()[label_index]
            temp_list = sig[init:end].append(current_label)
            data.append(temp_list)
        init = init + overlap
        end = end + overlap
    return data


def main():
    first_path = "/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/"
    for number in os.listdir(first_path):
        number_path = first_path + str(number)
        for second_number in os.listdir(number_path):
            second_number_path = number_path + "/" + str(second_number)
            for session in os.listdir(second_number_path):
                session_path = second_number_path + "/" + str(session)
                for file in os.listdir(session_path):
                    file_path = session_path + "/" + str(file)
                    if file_path.endswith('.tse'):
                        get_annotations(file_path, 0, 0)


new_ANN = nat.Annotations()
new_ANN.load("/Volumes/KESU/Datasets/TUH_EEG_Seizure_Corpus/v1.5.0/edf/train/01_tcp_ar/129/00012960/s002_2015_06_17/00012960_s002_t001.tse")
