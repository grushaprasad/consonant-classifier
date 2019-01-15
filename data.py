from __future__ import print_function
import librosa
import pickle, glob, re, sys
import csv
import torch
import os
import numpy as np
import random


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
                #print(filename)
                data = pickle.load(f)
                for item in data:
                    specs.append(item[0])
                    #print(item[0].shape)
                    labs.append(item[2])
    return([specs, labs])

def get_sets(train_path, test_path, split_method, split_proportion):

    all_train_specs, all_train_labs = load_specs(train_path)

    for item in all_train_labs:
        if item != 'D' and item != 'G':
            print('weird label before shuffling')

    all_train = list(zip(all_train_specs, all_train_labs))
    random.shuffle(all_train)
    all_train_specs, all_train_labs = zip(*all_train)
    for item in all_train_labs:
        if item != 'D' and item != 'G':
            print('weird label after shuffling')

    if split_method == 1:  #i.e by person
        split_ind = int(split_proportion * len(all_train))

        train_specs = all_train_specs[0:split_ind]
        train_labs = all_train_labs[0:split_ind]

        val_specs = all_train_specs[split_ind:]
        val_labs = all_train_labs[split_ind:]

    else:

        # each person has approx 100 words. So take words 1:80, 100:180 etc ..
        r = range(len(all_train_specs))
        train_inds = [ind for ind in r if ind % 100 < split_proportion*100]
        val_inds = [ind for ind in r if ind % 100 >= split_proportion*100]

        train_specs = [all_train_specs[i] for i in train_inds]
        train_labs = [all_train_labs[i] for i in train_inds]

        val_specs = [all_train_specs[i] for i in val_inds]
        val_labs = [all_train_labs[i] for i in val_inds]


        # Divide all_train into speakers. Then from each speaker take some proportion. 
        # split_ind = int(split_proportion * len(all_train_specs[1])) #assumes all speakers will have roughly equal observations
        
        # print(all_train_specs[1].shape)

        # train_specs = [x[0:split_ind] for x in all_train_specs]
        # #train_labs = [x[0:split_ind] for x in all_train_labs]
        # train_labs = [x for x in all_train_labs[0:split_ind]]
        # #train_labs = 

        # val_specs = [x[split_ind:] for x in all_train_specs]
        # #val_labs = [x[split_ind:] for x in all_train_labs]
        # val_labs = [x for x in all_train_labs[split_ind:]]

    # print(len(val_labs))
    # print(len(train_labs))
    # print(split_proportion)
    # # print(len(all_train_labs))
    # # print(type(all_train_labs[0]))
    # for item in val_labs:
    #     #print(item)
    #     if item != 'D' and item != 'G':
    #         print('weird label after shuffling')

    # for item in all_train_labs:
    #     print(item)

    test_specs, test_labs = load_specs(test_path)

    return(train_specs, train_labs, val_specs, val_labs, test_specs, test_labs)


def z_score(specs, mu, sd):
    print('__________________')
    z_scored = []
    for spec in specs:
        # print(spec.T.shape)
        # print('mu', mu.shape)
        # print('sd', sd.shape)
        z = ((spec - mu) / sd).T
        z_scored.append(z)
    return(z_scored)

# def zero_pad(spec, max_seq_length):
#     num_zeros = max_seq_length - spec.shape[1]
#     zeros = np.zeros([spec.shape[0], num_zeros]) 
#     zero_padded = np.hstack((spec, zeros))
#     return(zero_padded)

def truncate(spec, min_seq_length):
    return(spec[0:spec.shape[0], 0:min_seq_length]) # take all elements of first dimension and only 0:min_seq_length of second element. 

def get_mu_sd(specs):
    #print(type(specs))
    spec_all = np.concatenate(specs,0)
    #print(len(spec_all))
    mu = np.mean(spec_all,0)  # takes mean across filters
    # print(type(mu))
    # print(mu.shape)
    # for item in mu:
    #     print(item)
   

    sd = np.sqrt(np.var(spec_all, 0))
    return(mu,sd)

# def make_equal_lengths(specs, min_seq_length = -1, max_seq_length = -1):
#     if min_seq_length != -1:
#         return([truncate(spec, min_seq_length) for spec in specs])
#     elif max_seq_length != -1:
#         return([zero_pad(spec, max_seq_length) for spec in specs])
#     else:
#         raise Exception('Specify min or max seq_length')


def get_tensor(specs, labs):
    tensors_specs = []
    tensors_labs = []
    tensors = []
    for i,spec in enumerate(specs):
        curr_lab = labs[i]
        tensors_specs.append(torch.Tensor(spec))
        # print(curr_lab)
        # print('hello')
        tensors_labs.append(torch.LongTensor([lab_to_int[curr_lab]]))

    return(tensors_specs, tensors_labs)


class Melspectrogram(object):
    def __init__(self, train_path, test_path, split_proportion, split_method):
        
        train_specs, train_labs, val_specs, val_labs, test_specs, test_labs = get_sets(train_path, test_path, split_method, split_proportion)
        
        train_truncated = [truncate(spec, 10) for spec in train_specs]
        #print(train_truncated[0].shape)
        train_mu, train_sd = get_mu_sd(train_truncated)
        train_z_scored = z_score(train_truncated, train_mu, train_sd)

        self.train, self.train_labs = get_tensor(train_z_scored, train_labs)
       

        # get val data 
        val_truncated = [truncate(spec, 10) for spec in val_specs]
        val_z_scored = z_score(val_truncated, train_mu, train_sd)         
        self.val, self.val_labs = get_tensor(val_z_scored, val_labs)

        # get test data
        test_truncated = [truncate(spec, 10) for spec in test_specs]
        test_z_scored = z_score(test_truncated, train_mu, train_sd)       
        self.test, self.test_labs = get_tensor(test_z_scored, test_labs)



# all_train_dir = './ChodroffWilson2014/spectrograms/128/'
# test_dir = './SyntheticDG/spectrograms/'

# train_specs, train_labs, val_specs, val_labs, test_specs, test_labs = get_sets(all_train_dir, test_dir,  0, 0.8)

# print(len(train_specs))
# print(len(val_specs))

# print(len(train_specs[1]))
# print(len(val_specs[1]))




