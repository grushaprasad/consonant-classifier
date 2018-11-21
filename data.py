from __future__ import print_function
import librosa
import pickle
import csv
import torch
import os
import numpy as np
# directory = './audio_files/CV_stimuli/a/a_dg_CV/'


# dat = []

# for file in os.listdir(os.fsencode(directory)):
#     filename = directory + os.fsdecode(file)
#     y, sr = librosa.load(filename)
#     x = librosa.feature.melspectrogram(y=y, sr=sr)
#     dat.append(x)

# print(type(dat))
# # print(x[1])
# # print(x[6])
# print(len(dat))
# # print(x)
# y = torch.Tensor(dat)
# print(type(y))
# print(y.size())

lab_to_int = {
    'D': 0,
    'G': 1
}

class Spectrogram(object):
    def __init__(self, path):
        train_path = path + '/train/'
        valid_path = path + '/valid/'
        test_path = path + '/test/'

        train  = []
        for file in os.listdir(os.fsencode(train_path)):
            filename = train_path +  os.fsdecode(file)
            y, sr = librosa.load(filename)
            x = librosa.feature.melspectrogram(y=y, sr=sr)
            train.append([torch.Tensor(np.transpose(x)), torch.LongTensor([lab_to_int['D']])])  #CHANGE THIS TO ATTACH THE ACTUAL ANSWERS

        #self.train = [torch.Tensor(train), ['D']*20] # CHANGE THIS TO ATTACH THE ACTUAL ANSWERS
        self.train = train

        valid  = []
        for file in os.listdir(os.fsencode(valid_path)):
            filename = valid_path + os.fsdecode(file)
            y, sr = librosa.load(filename)
            x = librosa.feature.melspectrogram(y=y, sr=sr)
            valid.append(np.transpose(x))

        self.valid = [torch.Tensor(valid), ['D']*20] # CHANGE THIS TO ATTACH THE ACTUAL ANSWERS

        test  = []
        for file in os.listdir(os.fsencode(test_path)):
            filename = test_path + os.fsdecode(file)
            y, sr = librosa.load(filename)
            x = librosa.feature.melspectrogram(y=y, sr=sr)
            test.append([torch.Tensor(np.transpose(x)), torch.LongTensor([lab_to_int['D']])])

        self.test = test







