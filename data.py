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


def get_seqlength(dir_list):
    seq_lens = []
    for directory in dir_list:
        files = glob.glob(directory+'/*.pkl')
        for f in files:
            with open(f, 'rb') as curr_f:
                data = pickle.load(curr_f)
            for item in data:
                seq_lens.append(item[0].shape[1])
    return(max(seq_lens), min(seq_lens))


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

def zero_pad(specs, max_seq_length):
    num_zeros = max_seq_length - spec.shape[1]
    zeros = np.zeros([spec.shape[0], num_zeros]) 
    zero_padded = np.hstack((spec, zeros))
    return(zero_padded)

def truncate(spec, min_seq_length):
    return(spec[0:spec.shape[0], 0:min_seq_length]) # take all elements of first dimension and only 0:min_seq_length of second element. 

def get_mu_sd(specs):
    spec_all = np.concatenate(specs,1)
    mu = np.mean(spec_all,1)
    sd = np.sqrt(np.var(spec_all, 1))
    return(mu,sd)

def make_equal_lengths(specs, min_seq_length = -1, max_seq_length = -1):
    if min_seq_length != -1:
        return([truncate(spec, min_seq_length) for spec in specs])
    elif max_seq_length != -1:
        return([zero_pad(spec, min_seq_length) for spec in specs])
    else:
        raise Exception('Specify min or max seq_length')


def get_tensor(specs, labs):
    tensors_specs = []
    tensors_labs = []
    tensors = []
    for i,spec in enumerate(specs):
        curr_lab = labs[i]
        tensors_specs.append(torch.Tensor(spec))
        tensors_labs.append(torch.LongTensor([lab_to_int[curr_lab]]))

    return(tensors_specs, tensors_labs)


class Melspectrogram(object):
    def __init__(self, train_path, test_path, zero_or_truncate):
        
        self.max_seq_length, self.min_seq_length = get_seqlength(dir_list) 

        # z-score first
        if 0: 
            train_specs, train_labs, train_mu, train_sd = load_specs(train_path)
            train_mu, train_sd = get_mu_sd(train_specs)
            train_z_scored = z_score(train_specs, train_mu, train_sd)
            if zero_or_truncate == 'trunc':
                train_equal = make_equal_lengths(train_z_scored, min_seq_length = self.min_seq_length)
            else: 
                train_equal = make_equal_lengths(train_z_scored, max_seq_length = self.max_seq_length)

            self.train, self.train_labs = get_tensor(train_equal, train_labs)

            test_specs, test_labs, test_mu, test_sd = load_specs(test_path)
            test_z_scored = z_score(test_specs, train_mu, train_sd) # does this make sense??? Why? 
            if zero_or_truncate == 'trunc':
                test_equal = make_equal_lengths(test_z_scored, min_seq_length = self.min_seq_length)
            else: 
                test_equal = make_equal_lengths(test_z_scored, max_seq_length = self.max_seq_length)
            
            self.test, self.test_labs = get_tensor(test_equal, test_labs)

        # z-score last
        else: 
            train_specs, train_labs, train_mu, train_sd = load_specs(train_path)
            if zero_or_truncate == 'trunc':
                train_equal = make_equal_lengths(train_specs, min_seq_length = self.min_seq_length)
            else: 
                train_equal = make_equal_lengths(train_specs, max_seq_length = self.max_seq_length)

            train_mu, train_sd = get_mu_sd(train_equal)
            train_z_scored = z_score(train_equal, train_mu, train_sd)
            self.train, self.train_labs = get_tensor(train_z_scored, train_labs)

            test_specs, test_labs, test_mu, test_sd = load_specs(test_path)
            if zero_or_truncate == 'trunc':
                test_equal = make_equal_lengths(test_specs, min_seq_length = self.min_seq_length)
            else: 
                test_equal = make_equal_lengths(test_specs, max_seq_length = self.max_seq_length)

            test_z_scored = z_score(test_equal, train_mu, train_sd)
            self.test, self.test_labs = get_tensor(test_z_scored, test_labs)
