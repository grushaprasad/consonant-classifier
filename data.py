from __future__ import print_function
import librosa
import pickle, glob, re, sys
import csv
import torch
import os
import numpy as np
import random
import torch.nn.utils.rnn as rnn_utils


lab_to_int = {
    'D': 0,
    'G': 1
}

eps = 1.0e-8
power = 2.0


train = []


class Spec:
    def __init__(self,vals, seq_lens, labs, filenames):
        self.vals = vals
        self.seq_lens = seq_lens
        self.labs = labs
        self.filenames = filenames

        a = np.concatenate(self.vals, 1)
        a[a==0] = np.nan
        self.mu = np.nanmean(a,1)  
        self.sd = np.nanstd(a,1)


    def zscore(self, mu, sd):
        z_scored = []
        mu = mu[:,None]
        sd = sd[:,None]
        for val in self.vals:
            z = ((val - mu) / sd).T
            z_scored.append(z)
        self.zscored = z_scored



def get_mu_sd(specs):
    spec_all = np.concatenate(specs,1)
    mu = np.mean(spec_all,1) 
   
    sd = np.sqrt(np.var(spec_all, 1))
    return(mu,sd)

def get_seqlength(dir_list):
    seq_lens = []
    for directory in dir_list:
        files = glob.glob(directory+'/*.pkl')
        for f in files:
            with open(f, 'rb') as curr_f:
                data = pickle.load(curr_f)
            for item in data:
                seq_lens.append(item[1].shape[1])
    return(max(seq_lens), min(seq_lens))


def spectral_subtraction(spec, adaptor, alpha):  #function taken from Colin (sigproc.py)
    # print(type(spec))
    # print(type(adaptor))
    # print(type(alpha))
    spec, adaptor = np.exp(spec), np.exp(adaptor)
    spec_out = spec - alpha * adaptor

    #print(type(spec_out))

    spec_out = np.maximum(spec_out, 0.01*spec)
    spec_out = np.maximum(spec_out, 1.0e-8)
    spec_out = np.log(spec_out) 
    #print('adaptor', type(spec_out))

    return(spec_out)

def zero_pad(spec, max_seq_length):
    num_zeros = max_seq_length - spec.shape[1]
    zeros = np.zeros([spec.shape[0], num_zeros]) 
    zero_padded = np.hstack((spec, zeros))
    return(zero_padded)




def load_specs(names_file, adaptor = None, alpha = None):
    specs = []
    labs = []
    filenames = []
    with open(names_file) as f:
        files = f.read().splitlines()

    for f in files:
        with open(f, 'rb') as f:
            data = pickle.load(f)
            for item in data:  #len of item should be 6
                labs.append(item[3])
                filenames.append(item[0])
                
                if adaptor != None:
                    curr_spec = spectral_subtraction(item[1], adaptor, alpha)
                    #print('adaptor',type(adaptor))
                    #print('curr_spec', type(curr_spec))
                else:
                    curr_spec = item[1]
                specs.append(curr_spec)

    seq_length = [len(spec[0]) for spec in specs] #the length of spec will be same for every filter
    max_len = max(seq_length)

    specs = [zero_pad(spec, max_len) for spec in specs]

    # Shuffle training
    everything = list(zip(specs, seq_length, labs, filenames))
    random.shuffle(everything)
    specs, seq_length, labs, filenames = zip(*everything)

    spec = Spec(specs,seq_length, labs, filenames)

    return(spec)



def get_adaptor(adaptor_file):
    with open(adaptor_file, 'rb') as f:
        data = pickle.load(f)
    #print(data.shape)
    
    adaptor = np.mean(data, 1)   # this will change depending on the kind of averaging
    adaptor = adaptor[:,None]
    return(adaptor)



def get_sets(train_path, test_path, split_method, split_proportion, adaptor_file = None, alpha = 0.01):

    all_train = load_specs(train_path)

    if split_method == 1:  #i.e by person
        split_ind = int(split_proportion * len(all_train.vals))

        train_specs = all_train.vals[0:split_ind]
        train_seq_lens = all_train.seq_lens[0:split_ind]
        train_labs = all_train.labs[0:split_ind]
        train_files = all_train.filenames[0:split_ind]

        val_specs = all_train.vals[split_ind:]
        val_seq_lens = all_train.seq_lens[split_ind:]
        val_labs = all_train.labs[split_ind:]
        val_files = all_train.filenames[split_ind:]

    else: # each person has approx 100 words. So take words 1:80, 100:180 etc
        r = range(len(all_train.vals))
        train_inds = [ind for ind in r if ind % 100 < split_proportion*100]
        val_inds = [ind for ind in r if ind % 100 >= split_proportion*100]

        train_specs = [all_train.vals[i] for i in train_inds]
        train_seq_lens = [all_train.seq_lens[i] for i in train_inds]
        train_labs = [all_train.labs[i] for i in train_inds]
        train_files = [all_train.filenames[i] for i in train_inds]

        val_specs = [all_train.vals[i] for i in val_inds]
        val_seq_lens = [all_train.seq_lens[i] for i in val_inds]
        val_labs = [all_train.labs[i] for i in val_inds]
        val_files = [all_train.filenames[i] for i in val_inds]

    train = Spec(train_specs, train_seq_lens, train_labs, train_files)
    val = Spec(val_specs, val_seq_lens, val_labs, val_files)


    if adaptor_file:
        adaptor = get_adaptor(adaptor_file)
    else:
        adaptor = None

    test = load_specs(test_path, adaptor, alpha)

    return(train, val, test)


def get_tensor(specs, labs, filenames = 0):
    tensors_specs = []
    tensors_labs = []
    tensors_filenames = []
    for i,spec in enumerate(specs):
        curr_lab = labs[i]
        tensors_specs.append(torch.Tensor(spec))
        tensors_labs.append(torch.LongTensor([lab_to_int[curr_lab]]))
        if filenames != 0:
            tensors_filenames.append(filenames[i])  #not a tensor

    if filenames == 0: 
        return(tensors_specs, tensors_labs)
    else:
        return(tensors_specs, tensors_labs, tensors_filenames)


class Melspectrogram(object):
    def __init__(self, train_path, test_path, split_proportion, split_method, adaptor):
        
        train,val,test = get_sets(train_path, test_path, split_method, split_proportion, adaptor)
    
        train.zscore(train.mu, train.sd)
        self.train, self.train_labs = get_tensor(train.zscored, train.labs)
        self.train_files = train.filenames
        self.train_seq_lens = train.seq_lens

        val.zscore(train.mu, train.sd)        
        self.val, self.val_labs = get_tensor(val.zscored, val.labs)
        self.val_files = val.filenames
        self.val_seq_lens = val.seq_lens

        test.zscore(train.mu, train.sd)        
        self.test, self.test_labs = get_tensor(test.zscored, test.labs)
        self.test_files = test.filenames
        self.test_seq_lens = test.seq_lens

        #train_truncated = [truncate(spec, 10) for spec in train_specs]
        #train_mu, train_sd = get_mu_sd(train_truncated)
        #train_z_scored = z_score(train_truncated, train_mu, train_sd)

        #self.train, self.train_labs = get_tensor(train_z_scored, train_labs)
       
        # get val data 
        #val_truncated = [truncate(spec, 10) for spec in val_specs]
        #val_z_scored = z_score(val_truncated, train_mu, train_sd)
        
        

        # get test data
        #test_truncated = [truncate(spec, 10) for spec in test_specs]
        #test_z_scored = z_score(test_truncated, train_mu, train_sd)   
        # test_z_scored = z_score(test_specs, train_mu, train_sd)
        # self.test, self.test_labs, self.test_files = get_tensor(test_z_scored, test_labs, test_files)


#print(dir_list[0])
#print(get_seqlength(dir_list))
# all_train_dir = './ChodroffWilson2014/spectrograms/128/'
# test_dir = './SyntheticDG/spectrograms/'

# train = './data/train.txt'
# test = './data/test.txt'

# train_specs, train_labs, train_files, val_specs, val_labs, val_files, test_specs, test_labs, test_files = get_sets(train, test,  0, 0.8)
# train_mu, train_sd = get_mu_sd(train_specs)

#print(train_mu)

# print(len(train_specs))
# print(len(val_specs))

# print(len(train_specs[1]))
# print(len(val_specs[1]))

x = load_specs('./data/train.txt')


"""

To do: 
1. Make batches for test and validation 

"""



