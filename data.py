from __future__ import print_function
import librosa
import pickle
import csv
import torch
import os
import numpy as np
import re


lab_to_int = {
    'D': 0,
    'G': 1
}

class Melspectrogram(object):
    def __init__(self, train_path, test_path):

        train  = []
        for file in os.listdir(os.fsencode(train_path)):
            filename = train_path +  os.fsdecode(file)
            if len(re.findall('\d+', filename)) > 0:  #ignores other files
                filenum = int(re.findall('\d+', filename)[0])
                if filenum < 6:
                    lab = 'D'
                else:
                    lab = 'G'
                y, sr = librosa.load(filename)
                x = librosa.feature.melspectrogram(y=y, sr=sr)
                train.append([torch.Tensor(np.transpose(x)), torch.LongTensor([lab_to_int[lab]])]) 

        self.train = train

        test  = []
        for file in os.listdir(os.fsencode(test_path)):
            filename = test_path + os.fsdecode(file)
            if len(re.findall('\d+', filename)) > 0:  #ignores other files
                filenum = int(re.findall('\d+', filename)[0])
                if filenum < 11:
                    lab = 'D'
                else:
                    lab = 'G'
                y, sr = librosa.load(filename)
                x = librosa.feature.melspectrogram(y=y, sr=sr)
                test.append([torch.Tensor(np.transpose(x)), torch.LongTensor([lab_to_int[lab]])])

        self.test = test







