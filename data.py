from __future__ import print_function
import librosa
import pickle, glob, re, sys
import csv
import torch
import os
import numpy as np
import random
import torch.nn.utils.rnn as rnn_utils


# lab_to_int = {
#     'D': 0,
#     'G': 1
# }

eps = 1.0e-8

combined_to_int = {
    'DAA': 0,
    'DAE': 1,
    'DIY': 2,
    'DUW': 3,
    'GAA': 4,
    'GAE': 5,
    'GIY': 6,
    'GUW': 7
}

vowel_to_int = {
    'AA': 0,
    'AE': 1,
    'IY': 2,
    'UW': 3
}

cons_to_int = {
    'D': 0,
    'G':1
}

eps = 1.0e-8
power = 2.0


train = []


class Spec:
    def __init__(self,vals, seq_lens, cons_labs, vowel_labs, combined_labs, filenames):
        self.vals = vals
        self.seq_lens = seq_lens
        self.filenames = filenames
        self.cons_labs = cons_labs
        self.vowel_labs = vowel_labs
        self.combined_labs = combined_labs

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


def get_adaptor(adaptor_file):
    with open(adaptor_file, 'rb') as f:
        data = pickle.load(f)
    # print(type(data))
    # print(data.shape)
    
    adaptor = np.mean(data, 1)   # this will change depending on the kind of averaging
    adaptor = adaptor[:,None]
    return(adaptor)


def subtract_precursor(spec, adaptor_file, alpha):  #function taken from Colin (sigproc.py)
    # print(type(spec))
    # print(type(adaptor))
    # print(type(alpha))
    if adaptor_file != 'NA':
        adaptor = get_adaptor(adaptor_file)
        #print(adaptor, adaptor_file)
        spec, adaptor = np.exp(spec), np.exp(adaptor)
        spec_out = spec - (alpha * adaptor)
        spec_out = np.maximum(spec_out, 0)  #gets rid of negative numbers
        spec_out = np.log(spec_out + eps)

        # This is stuff that Colin had
        # spec_out = np.maximum(spec_out, 0.01*spec)
        # spec_out = np.maximum(spec_out, 1.0e-8)
        #print('adaptor', type(spec_out))

        return(spec_out)

    else:
        return(spec)

def zero_pad(spec, max_seq_length):
    num_zeros = max_seq_length - spec.shape[1]
    zeros = np.zeros([spec.shape[0], num_zeros]) 
    zero_padded = np.hstack((spec, zeros))
    return(zero_padded)


def load_specs(names_file, adaptor_file = None, alpha = None):
    specs = []
    combined_labs = []
    cons_labs = []
    vowel_labs = []
    filenames = []
    lab_counts = {
        'DAA': 0,
        'DAE': 0,
        'DIY': 0,
        'DUW': 0,
        'GAA': 0,
        'GAE': 0,
        'GIY': 0,
        'GUW': 0
    }
    with open(names_file) as f:
        files = f.read().splitlines()

    for f in files:
        with open(f, 'rb') as f:
            data = pickle.load(f)
            for item in data:  #len of item should be 6

                # lab = item[3]+item[4]
                # lab_counts[lab]+=1

                cons_labs.append(item[3])
                vowel_labs.append(item[4])
                combined_labs.append(item[3]+item[4])
                filenames.append(item[0])
                
                if adaptor_file:
                    curr_spec = subtract_precursor(item[1], adaptor_file, alpha)
                    #print('adaptor',type(adaptor))
                    #print('curr_spec', type(curr_spec))
                else:
                    curr_spec = item[1]
                specs.append(curr_spec)

    seq_length = [len(spec[0]) for spec in specs] #the length of spec will be same for every filter
    max_len = max(seq_length)
    #print(max_len)
    max_len = 107

    specs = [zero_pad(spec, max_len) for spec in specs]

    # Shuffle training
    everything = list(zip(specs, seq_length, cons_labs, vowel_labs, combined_labs, filenames))
    random.shuffle(everything)
    specs, seq_length, cons_labs, vowel_labs, combined_labs, filenames = zip(*everything)

    spec = Spec(specs,seq_length, cons_labs, vowel_labs, combined_labs, filenames)
    #print(names_file,lab_counts)
    return(spec)




def get_sets(train_path, test_path, split_method, split_proportion, adaptor_file, alpha = 0.1):
    #print('split prop', split_proportion)
    all_train = load_specs(train_path)

    if split_method == 1:  #i.e by person
        split_ind = int(split_proportion * len(all_train.vals))

        train_specs = all_train.vals[0:split_ind]
        train_seq_lens = all_train.seq_lens[0:split_ind]
        train_cons_labs = all_train.cons_labs[0:split_ind]
        train_vowel_labs = all_train.vowel_labs[0:split_ind]
        train_combined_labs = all_train.combined_labs[0:split_ind]
        train_files = all_train.filenames[0:split_ind]

        val_specs = all_train.vals[split_ind:]
        val_seq_lens = all_train.seq_lens[split_ind:]
        val_cons_labs = all_train.cons_labs[split_ind:]
        val_vowel_labs = all_train.vowel_labs[split_ind:]
        val_combined_labs = all_train.combined_labs[split_ind:]
        val_files = all_train.filenames[split_ind:]

    else: # each person has approx 100 words. So take words 1:80, 100:180 etc
        r = range(len(all_train.vals))
        train_inds = [ind for ind in r if ind % 100 < split_proportion*100]
        val_inds = [ind for ind in r if ind % 100 >= split_proportion*100]

        train_specs = [all_train.vals[i] for i in train_inds]
        train_seq_lens = [all_train.seq_lens[i] for i in train_inds]
        train_cons_labs = [all_train.cons_labs[i] for i in train_inds]
        train_vowel_labs = [all_train.vowel_labs[i] for i in train_inds]
        train_combined_labs = [all_train.combined_labs[i] for i in train_inds]
        train_files = [all_train.filenames[i] for i in train_inds]

        val_specs = [all_train.vals[i] for i in val_inds]
        val_seq_lens = [all_train.seq_lens[i] for i in val_inds]
        val_cons_labs = [all_train.cons_labs[i] for i in val_inds]
        val_vowel_labs = [all_train.vowel_labs[i] for i in val_inds]
        val_combined_labs = [all_train.combined_labs[i] for i in val_inds]
        val_files = [all_train.filenames[i] for i in val_inds]

    train = Spec(train_specs, train_seq_lens, train_cons_labs, train_vowel_labs, train_combined_labs, train_files)
    val = Spec(val_specs, val_seq_lens, val_cons_labs, val_vowel_labs, val_combined_labs,val_files)


    # if adaptor_file != 'NA':
    #     adaptor = get_adaptor(adaptor_file)
    # else:
    #     adaptor = None

    test = load_specs(test_path)
    test_subtracted = load_specs(test_path, adaptor_file, alpha)

    return(train, val, test, test_subtracted)


def get_tensor(specs, cons_labs, vowel_labs, combined_labs, filenames = 0):
    tensors_specs = []
    tensors_cons_labs = []
    tensors_vowel_labs = []
    tensors_combined_labs = []
    tensors_filenames = []
    for i,spec in enumerate(specs):
        curr_cons_lab = cons_labs[i]
        curr_vowel_lab = vowel_labs[i]
        curr_combined_lab = combined_labs[i]

        if torch.cuda.is_available():
            tensors_cons_labs.append(torch.cuda.LongTensor([cons_to_int[curr_cons_lab]]))
            tensors_vowel_labs.append(torch.cuda.LongTensor([vowel_to_int[curr_vowel_lab]]))
            tensors_combined_labs.append(torch.cuda.LongTensor([combined_to_int[curr_combined_lab]]))
            tensors_specs.append(torch.cuda.FloatTensor(spec))

        else:
            tensors_cons_labs.append(torch.LongTensor([cons_to_int[curr_cons_lab]]))
            tensors_vowel_labs.append(torch.LongTensor([vowel_to_int[curr_vowel_lab]]))
            tensors_combined_labs.append(torch.LongTensor([combined_to_int[curr_combined_lab]]))
            tensors_specs.append(torch.Tensor(spec))
        
        
        if filenames != 0:
            tensors_filenames.append(filenames[i])  #not a tensor

    if filenames == 0: 
        return(tensors_specs, tensors_cons_labs, tensors_vowel_labs, tensors_combined_labs)
    else:
        return(tensors_specs, tensors_cons_labs, tensors_vowel_labs, tensors_combined_labs, tensors_filenames)


class Melspectrogram(object):
    def __init__(self, train_path, test_path, split_proportion, split_method, adaptor_file):
        
        train,val,test,test_subtracted = get_sets(train_path, test_path, split_method, split_proportion, adaptor_file)
    
        train.zscore(train.mu, train.sd)
        self.train, self.train_cons_labs, self.train_vowel_labs, self.train_combined_labs = get_tensor(train.zscored, train.cons_labs, train.vowel_labs, train.combined_labs)
        self.train_files = train.filenames
        self.train_seq_lens = train.seq_lens

        val.zscore(train.mu, train.sd)        
        self.val, self.val_cons_labs, self.val_vowel_labs, self.val_combined_labs = get_tensor(val.zscored, val.cons_labs, val.vowel_labs, val.combined_labs)
        self.val_files = val.filenames
        self.val_seq_lens = val.seq_lens

        test.zscore(train.mu, train.sd)        
        self.test, self.test_cons_labs, self.test_vowel_labs, self.test_combined_labs = get_tensor(test.zscored, test.cons_labs, test.vowel_labs, test.combined_labs)
        self.test_files = test.filenames
        self.test_seq_lens = test.seq_lens

        test_subtracted.zscore(train.mu, train.sd)        
        self.test_subtracted, self.test_subtracted_cons_labs, self.test_subtracted_vowel_labs, self.test_subtracted_combined_labs = get_tensor(test_subtracted.zscored, test_subtracted.cons_labs, test_subtracted.vowel_labs, test_subtracted.combined_labs)
        self.test_subtracted_files = test_subtracted.filenames
        self.test_subtracted_seq_lens = test_subtracted.seq_lens

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

#x = load_specs('./data/train.txt')




