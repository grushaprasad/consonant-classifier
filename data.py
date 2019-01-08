from __future__ import print_function
import librosa
import pickle, glob, re, sys
import csv
import torch
import os
import numpy as np


lab_to_int = {
    'D': 0,
    'G': 1
}

eps = 1.0e-8
power = 2.0


train = []
dir_list = ['./ChodroffWilson2014/spectrograms/64/',
 './ChodroffWilson2014/spectrograms/128/',
 './SyntheticDG/spectrograms/' ]


def get_max_seqlength(dir_list):
    seq_lens = []
    for directory in dir_list:
        files = glob.glob(directory+'/*.pkl')
        for f in files:
            with open(f, 'rb') as curr_f:
                data = pickle.load(curr_f)
            for item in data:
                seq_lens.append(item[0].shape[1])
    return(max(seq_lens))



def load_specs(directory):
    specs = []
    labs = []
    for file in os.listdir(os.fsencode(directory)):
        filename = directory +  os.fsdecode(file)
        if filename.endswith(".pkl"):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                for item in data:
                    specs.append(item[0])
                    labs.append(item[1])
    spec_all = np.concatenate(specs,1)
    mu = np.mean(spec_all,1)
    sd = np.sqrt(np.var(spec_all, 1))
    return([specs, labs, mu, sd])


def z_score(specs, mu, sd):
    z_scored = []
    for spec in specs:
        z = ((spec.T - mu) / sd).T
        z_scored.append(z)
    return(z_scored)

def get_tensor(specs, labs, max_seq_length):
    tensors = []
    for i,spec in enumerate(specs):
        curr_lab = labs[i]
        num_zeros = max_seq_length - spec.shape[1]
        zeros = np.zeros([spec.shape[0], num_zeros]) 
        zero_padded = np.hstack((spec, zeros))
        tensors.append([torch.Tensor(zero_padded), torch.LongTensor([lab_to_int[curr_lab]])])
    return(tensors)



class Melspectrogram(object):
    def __init__(self, train_path, test_path):
        
        self.max_seq_length = get_max_seqlength(dir_list) 

        train_specs, train_labs, train_mu, train_sd = load_specs(train_path)
        train_z_scored = z_score(train_specs, train_mu, train_sd)
        self.train = get_tensor(train_z_scored, train_labs, self.max_seq_length)

        test_specs, test_labs, test_mu, test_sd = load_specs(test_path)
        test_z_scored = z_score(test_specs, train_mu, train_sd) # does this make sense??? Why? 
        self.test = get_tensor(test_z_scored, test_labs, self.max_seq_length)





        





