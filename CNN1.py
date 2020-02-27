import pyedflib
import os
import matplotlib.pyplot as plt
files_path_train = "/Volumes/KESU/Datasets/TUH EEG Seizure Corpus/v1.5.0/edf/train"
files_path_dev = "/Volumes/KESU/Datasets/TUH EEG Seizure Corpus/v1.5.0/edf/dev_set"

filename = "/Volumes/KESU/Datasets/TUH EEG Seizure Corpus/v1.5.0/edf/train/01_tcp_ar/000/00000077/s003_2010_01_21/00000077_s003_t000.edf"

##reading a edf signal
f = pyedflib.EdfReader(filename)
channel_0 = f.readSignal(0)
plt.figure()
plt.plot(channel_0)
plt.show()
print(f.getHeader())
#print(signals)
#print(signal_headers)
#print(signals)
