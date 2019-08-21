import torch 
import torch.nn as nn
import argparse
import torch.nn.utils.rnn as rnn_utils
import pickle

import data
import model
import random
import re
import glob
import os

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='PyTorch consonant classifier')

parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')

parser.add_argument('--modelpath', type=str, default='./models/combined/',
                    help='path to save the final model')

parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--traindir', type=str, default='./data/train.txt',
                    help='directory with training spectrograms')

parser.add_argument('--testdir', type=str, default='./data/test.txt',
                    help='directory with test spectrograms')

parser.add_argument('--datdir', type=str, default='./csv/',
                    help='directory with test spectrograms')

parser.add_argument('--testmodel', type=str, default='NA',
                    help='name of pre-trained model you want to test')

parser.add_argument('--adaptor_dir', type=str, default='./precursors/unpadded/20mels/',
                    help='file path of adaptor that will get subtracted from test')

parser.add_argument('--nmels', type=int, default=20,
                    help='number of mel features')

parser.add_argument('--bsz', type=int, default=10,
                    help='batch_size')

parser.add_argument('--split_prop', type=float, default=0.8,
                    help='proportion of training and validation split')

parser.add_argument('--split_method', type=int, default=1,
                    help='split by person: 1, split by tokens: 0')

parser.add_argument('--epoch_prop', type=float, default=0.8,
                    help='Value between 0 and 1. Determines what percent of training data network sees per epoch')

parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')

parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units')

parser.add_argument('--ncons', type=int, default=2,
                    help='number of consonants')

parser.add_argument('--nvowels', type=int, default=4,
                    help='number of vowels')

parser.add_argument('--classification', type=int, default=1,
                    help='combined cons and vowel: 1, split cons and vowel: 0')


parser.add_argument('--seed', type=int, default=17,
                    help='random seed')


args = parser.parse_args()

# DEFINE PARAMETERS
# Input features
input_size = args.nmels  #number of features per time step 
num_classes = 8

#Hyper-parameters
num_epochs = args.epochs
num_layers = args.nlayers
batch_size = args.bsz
hidden_size = args.nhid
learning_rate = args.lr
num_cons = args.ncons
num_vowels = args.nvowels
num_combined = num_cons*num_vowels

#sequence_length = 50
#sequence_length = 10
sequence_length = 107

random.seed(args.seed)

modelname = args.modelpath + 'model' + str(input_size) + '-' + str(learning_rate) + '-' + str(num_epochs) + '-' +  str(num_layers) + '-' + str(hidden_size) + '-' + str(args.seed) + '.pt'

if args.testmodel != 'NA':
    modelpath = args.testmodel 
else:
    modelpath = modelname

trainpath = modelpath.replace('.pt', '.tr')
testpath = modelpath.replace('.pt', '.test')
valpath = modelpath.replace('.pt', '.val')


def flatten(l):
    return([item for sublist in l for item in sublist])

def split(l, n):
    # modified from: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115
    k,m = divmod(len(l), round(len(l)/n))
    new = [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    return(new)

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def get_batch(l, bsz):
    new = list(chunks(l, bsz))
    if len(new[-1]) != bsz:
        new = new[:-1]
    return(new)


def get_random_sample(batched_dat, prop):
    split_ind = int(prop * len(batched_dat[0]))
    shuffled = [random.sample(batch, len(batch)) for batch in batched_dat] 
    subset = [batch[0:split_ind] for batch in shuffled]
    return(subset)


def order_set(seq_lens, dat, cons_labs, vowel_labs, combined_labs, files):
    for i in range(len(seq_lens)):
        zipped = zip(list(seq_lens[i]), dat[i], cons_labs[i], vowel_labs[i], combined_labs[i], files[i])
        ordered = reversed(sorted(zipped,  key = lambda x: x[0]))
        seq_lens[i], dat[i], cons_labs[i], vowel_labs[i], combined_labs[i], files[i] = zip(*ordered)
 

# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SETUP
# Retrieve/ create training, val and test data and model

if args.testmodel != 'NA':  #i.e. there is an existing test model

    with open(trainpath, 'rb') as f:
        train_seq_lens, train_data, train_cons_labs, train_vowel_labs, train_combined_labs, train_filenames = pickle.load(f)

    with open(valpath, 'rb') as f:
        val_seq_lens, val_data, val_cons_labs, val_vowel_labs, val_combined_labs, val_filenames = pickle.load(f)

    with open(testpath, 'rb') as f:
        test_seq_lens, test_data, test_cons_labs, test_vowel_labs, test_combined_labs, test_filenames = pickle.load(f)

    with open(args.testmodel, 'rb') as f:
        model = torch.load(f)

else:
    # print('hello')

    dat = data.Melspectrogram(args.traindir, args.testdir, args.split_prop, args.split_method, 'NA')

    train_data = get_batch(dat.train, batch_size)
    train_cons_labs = get_batch(dat.train_cons_labs, batch_size)
    train_vowel_labs = get_batch(dat.train_vowel_labs, batch_size)
    train_combined_labs = get_batch(dat.train_combined_labs, batch_size)
    train_seq_lens = get_batch(dat.train_seq_lens, batch_size)
    train_filenames = get_batch(dat.train_files, batch_size)

    order_set(train_seq_lens, train_data, train_cons_labs, train_vowel_labs, train_combined_labs, train_filenames)

    with open(trainpath, 'wb') as f:
        pickle.dump([train_seq_lens, train_data, train_cons_labs, train_vowel_labs, train_combined_labs, train_filenames], f)

    val_data = get_batch(dat.val, batch_size)
    val_cons_labs = get_batch(dat.val_cons_labs, batch_size)
    val_vowel_labs = get_batch(dat.val_vowel_labs, batch_size)
    val_combined_labs = get_batch(dat.val_combined_labs, batch_size)
    val_seq_lens = get_batch(dat.val_seq_lens, batch_size)
    val_filenames = get_batch(dat.val_files, batch_size)

    order_set(val_seq_lens, val_data, val_cons_labs, val_vowel_labs, val_combined_labs, val_filenames)

    with open(valpath, 'wb') as f:
        pickle.dump([val_seq_lens, val_data, val_cons_labs, val_vowel_labs, val_combined_labs, val_filenames], f)

    test_data = get_batch(dat.test, batch_size)
    test_cons_labs = get_batch(dat.test_cons_labs, batch_size)
    test_vowel_labs = get_batch(dat.test_vowel_labs, batch_size)
    test_combined_labs = get_batch(dat.test_combined_labs, batch_size)
    test_seq_lens = get_batch(dat.test_seq_lens, batch_size)
    test_filenames = get_batch(dat.test_files, batch_size)

    order_set(test_seq_lens, test_data, test_cons_labs, test_vowel_labs, test_combined_labs, test_filenames)

    with open(testpath, 'wb') as f:
        pickle.dump([test_seq_lens, test_data, test_cons_labs, test_vowel_labs, test_combined_labs, test_filenames], f)




    #print(train_data)
    #print(type(dat.train))
    # Initialize model
    model = model.BiRNN(input_size, hidden_size, num_layers, num_cons, num_vowels, args.classification).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print(train_data[0].size())
# print(test_data[0].size())
# print(val_data[0].size())
# print(len(dat.train))
# print(len(dat.val))
# # print('t', len(train_data[0]))
# # print('v', len(val_data[0]))
# print(train_data[0][1].size())
# print(test_data[0][1].size())
# print(val_data[0][1].size())
# print(len(train_labs))
# print(test_labs[0])

# # TRAINING and VALIDATION

if args.testmodel == 'NA': #i.e. there isn't a pre-trained model
    ## Train
    total_step = len(train_data)
    for epoch in range(num_epochs):
        subset = get_random_sample(train_data, args.epoch_prop)
        for i,batch in enumerate(train_data): 
            grouped = torch.stack(batch)  #combines list of tensors into a dimension

            sound = grouped.reshape(batch_size, sequence_length, input_size).to(device)
            

            if torch.cuda.is_available():
                cons_label = torch.cuda.LongTensor(train_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
                vowel_label = torch.cuda.LongTensor(train_vowel_labs[i]).to(device)
                combined_label = torch.cuda.LongTensor(train_combined_labs[i]).to(device)
                seq_len = torch.cuda.LongTensor(train_seq_lens[i]).to(device)
            else:
                cons_label = torch.LongTensor(train_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
                vowel_label = torch.LongTensor(train_vowel_labs[i]).to(device)
                combined_label = torch.LongTensor(train_combined_labs[i]).to(device)
                seq_len = torch.LongTensor(train_seq_lens[i]).to(device)

            

            #forward pass
            output = model(sound, seq_len)

            if args.classification == 1: #i.e. combined vowel and cons classification
                loss = criterion(output, combined_label)
            else:
                cons_loss = criterion(output[0], cons_label)
                vowel_loss = criterion(output[1], vowel_label)
                loss = cons_loss + 0.2*vowel_loss
                #loss = cons_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))


    with open(modelpath, 'wb') as f:
        torch.save(model, f)

    
# ## TEST

## Test on training data
with torch.no_grad():
    correct = 0
    correct_cons = 0
    correct_vowel = 0
    total = 0
    total_cons = 0
    total_vowel = 0

    for i, batch in enumerate(train_data):
        grouped = torch.stack(batch)
        sound = grouped.reshape(batch_size, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
        
        if torch.cuda.is_available():
            cons_label = torch.cuda.LongTensor(train_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.cuda.LongTensor(train_vowel_labs[i]).to(device)
            combined_label = torch.cuda.LongTensor(train_combined_labs[i]).to(device)
            seq_len = torch.cuda.LongTensor(train_seq_lens[i]).to(device)
        else:
            cons_label = torch.LongTensor(train_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.LongTensor(train_vowel_labs[i]).to(device)
            combined_label = torch.LongTensor(train_combined_labs[i]).to(device)
            seq_len = torch.LongTensor(train_seq_lens[i]).to(device)

        
        output = model(sound, seq_len)

        if args.classification == 1:
            _, predicted = torch.max(output.data, 1)
            total += combined_label.size(0)
            correct += (predicted == combined_label).sum().item()
        
        else:
            _, predicted_cons = torch.max(output[0].data, 1)
            _, predicted_vowel = torch.max(output[1].data, 1)
            total_cons += cons_label.size(0)
            total_vowel += vowel_label.size(0)
            correct_cons += (predicted_cons == cons_label).sum().item()
            correct_vowel += (predicted_vowel == vowel_label).sum().item()


    if total != 0 and args.classification==1:
        print('Test Accuracy on training data (vowels, cons combined) {} %'.format(100 * correct / total)) 
    else:
        print('Test Accuracy on training data: cons {} %'.format(100 * correct_cons / total_cons))
        print('Test Accuracy on training data: vowels {} %'.format(100 * correct_vowel / total_vowel))

##  Test on held out natural speech
with torch.no_grad():
    correct = 0
    correct_cons = 0
    correct_vowel = 0
    total = 0
    total_cons = 0
    total_vowel = 0

    for i, batch in enumerate(val_data):
        grouped = torch.stack(batch)
        sound = grouped.reshape(batch_size, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
        if torch.cuda.is_available():
            cons_label = torch.cuda.LongTensor(val_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.cuda.LongTensor(val_vowel_labs[i]).to(device)
            combined_label = torch.cuda.LongTensor(val_combined_labs[i]).to(device)
            seq_len = torch.cuda.LongTensor(val_seq_lens[i]).to(device)
        else:
            cons_label = torch.LongTensor(val_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.LongTensor(val_vowel_labs[i]).to(device)
            combined_label = torch.LongTensor(val_combined_labs[i]).to(device)
            seq_len = torch.LongTensor(val_seq_lens[i]).to(device)
        
        output = model(sound, seq_len)

        if args.classification == 1:
            _, predicted = torch.max(output.data, 1)
            total += combined_label.size(0)
            correct += (predicted == combined_label).sum().item()
        
        else:
            _, predicted_cons = torch.max(output[0].data, 1)
            _, predicted_vowel = torch.max(output[1].data, 1)
            total_cons += cons_label.size(0)
            total_vowel += vowel_label.size(0)
            correct_cons += (predicted_cons == cons_label).sum().item()
            correct_vowel += (predicted_vowel == vowel_label).sum().item()


    if total != 0 and args.classification==1:
        print('Test Accuracy on held out data (vowels, cons combined) {} %'.format(100 * correct / total)) 
    else:
        print('Test Accuracy on held out data: cons {} %'.format(100 * correct_cons / total_cons))
        print('Test Accuracy on held out data: vowels {} %'.format(100 * correct_vowel / total_vowel))


# Test on continuum
with torch.no_grad():

    correct = 0
    total = 0
    filenum_dict = dict((el, []) for el in range(1,21))
    filename_dict = dict((el, 0) for el in flatten(test_filenames))
    print(len(filename_dict))
    probs_dict = {0: [], 1: []}

    for i, batch in enumerate(test_data):
        grouped = torch.stack(batch)
        sound = grouped.reshape(batch_size, sequence_length, input_size).to(device) 
        
        if torch.cuda.is_available():
            cons_label = torch.cuda.LongTensor(test_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.cuda.LongTensor(test_vowel_labs[i]).to(device)
            combined_label = torch.cuda.LongTensor(test_combined_labs[i]).to(device)
            seq_len = torch.cuda.LongTensor(test_seq_lens[i]).to(device)
        else:
            cons_label = torch.LongTensor(test_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
            vowel_label = torch.LongTensor(test_vowel_labs[i]).to(device)
            combined_label = torch.LongTensor(test_combined_labs[i]).to(device)
            seq_len = torch.LongTensor(test_seq_lens[i]).to(device)

        
        output = model(sound, seq_len)
        filenum = [int(re.findall('\d+', x)[0]) for x in test_filenames[i]]
        filename = [x for x in test_filenames[i]]
        
        sm = torch.nn.Softmax(dim=1)
        if args.classification==1:
            all_probs = sm(output)
            probs = [[sum(x[0:4]), sum(x[4:8])] for x in all_probs]
            # print(all_probs)
            # print(len(probs))

        else:
            probs = sm(output[0])
            # print(probs)

        curr = []
        for i,p in enumerate(probs):
            filename_dict[filename[i]] = round(float(p[1].item()), 3) #change to p[0] for % D


    dat_path = '%s%s/'%(args.datdir,args.classification)
    print(modelpath)
    print(dat_path)

    if not os.path.exists(dat_path):
        os.makedirs(dat_path)

    with open('%s%s_pre.csv'%(dat_path,os.path.basename(modelpath)[:-3]), 'w') as f:
        f.write('fname, filenum, vowel, adaptor, prob_G \n')
        for key in filename_dict:
            filenum = int(re.findall('\d+', key)[0])
            f_name = os.path.basename(key)
            if f_name[1] == '_':
                vowel = f_name[0:1]
            else:
                vowel = f_name[0:2]

            f.write('%s,%s,%s,%s,%s \n'%(f_name, filenum, vowel, 'no-adapt' , filename_dict[key]))

    # for i,key in enumerate(sorted(filename_dict)):  # Print by filenum
    #     if i%20==0:
    #         print('-----------------------')
    #     print('%s: %s'%(key,filename_dict[key]))


# Test on continuum after subtraction
#print('TESTING AFTER SUBTRACTION')
# adaptor_list = ['./precursors/unpadded/20mels/midhigh.pkl',
#     './precursors/unpadded/20mels/highmid.pkl',
#     './precursors/unpadded/20mels/midlow.pkl', 
#     './precursors/unpadded/20mels/lowmid.pkl']

adaptor_list = glob.glob(args.adaptor_dir+'*.pkl')

with open('%s%s_post.csv'%(dat_path,os.path.basename(modelpath)[:-3]), 'w') as f:
    f.write('fname, filenum, vowel, adaptor, prob_G \n')

    for adaptor in adaptor_list:
        print(adaptor)
        with torch.no_grad():
            dat = data.Melspectrogram(args.traindir, args.testdir, args.split_prop, args.split_method, adaptor)

            subtracted_test_data = get_batch(dat.test_subtracted, batch_size)
            subtracted_test_cons_labs = get_batch(dat.test_subtracted_cons_labs, batch_size)
            subtracted_test_vowel_labs = get_batch(dat.test_subtracted_vowel_labs, batch_size)
            subtracted_test_combined_labs = get_batch(dat.test_subtracted_combined_labs, batch_size)
            subtracted_test_seq_lens = get_batch(dat.test_subtracted_seq_lens, batch_size)
            subtracted_test_filenames = get_batch(dat.test_subtracted_files, batch_size)

            order_set(subtracted_test_seq_lens, subtracted_test_data, subtracted_test_cons_labs, subtracted_test_vowel_labs, subtracted_test_combined_labs, subtracted_test_filenames)

            correct = 0
            total = 0
            filenum_dict = dict((el, []) for el in range(1,21))
            filename_dict = dict((el, 0) for el in flatten(subtracted_test_filenames))
            probs_dict = {0: [], 1: []}

            for i, batch in enumerate(subtracted_test_data):
                grouped = torch.stack(batch)
                sound = grouped.reshape(batch_size, sequence_length, input_size).to(device) 

                if torch.cuda.is_available():
                    cons_label = torch.cuda.LongTensor(subtracted_test_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
                    vowel_label = torch.cuda.LongTensor(subtracted_test_vowel_labs[i]).to(device)
                    combined_label = torch.cuda.LongTensor(subtracted_test_combined_labs[i]).to(device)
                    seq_len = torch.cuda.LongTensor(subtracted_test_seq_lens[i]).to(device)
                else:
                    cons_label = torch.LongTensor(subtracted_test_cons_labs[i]).to(device)  #labels need to be a tensor of tensors
                    vowel_label = torch.LongTensor(subtracted_test_vowel_labs[i]).to(device)
                    combined_label = torch.LongTensor(subtracted_test_combined_labs[i]).to(device)
                    seq_len = torch.LongTensor(subtracted_test_seq_lens[i]).to(device)
                
                output = model(sound, seq_len)
                filenum = [int(re.findall('\d+', x)[0]) for x in subtracted_test_filenames[i]]
                filename = [x for x in subtracted_test_filenames[i]]
                
                sm = torch.nn.Softmax(dim=1)
                if args.classification==1:
                    all_probs = sm(output)
                    probs = [[sum(x[0:4]), sum(x[4:8])] for x in all_probs]
                    # print(all_probs)
                    # print(len(probs))

                else:
                    probs = sm(output[0])
                    # print(probs)

                curr = []
                for i,p in enumerate(probs):
                    filename_dict[filename[i]] = round(float(p[1].item()), 3)  #change to p[0] for % D


            for key in filename_dict:
                filenum = int(re.findall('\d+', key)[0])
                f_name = os.path.basename(key)
                if f_name[1] == '_':
                    vowel = f_name[0:1]
                else:
                    vowel = f_name[0:2]

                f.write('%s,%s,%s,%s,%s \n'%(f_name, filenum, vowel, os.path.basename(adaptor), filename_dict[key]))


            # for i,key in enumerate(sorted(filename_dict)):  # Print by filenum
            #     if i%20==0:
            #         print('-----------------------')
            #     print('%s: %s'%(key,filename_dict[key]))

