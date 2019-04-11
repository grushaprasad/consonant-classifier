import torch 
import torch.nn as nn
import argparse
import torch.nn.utils.rnn as rnn_utils


import data
import model
import random
import re

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='PyTorch consonant classifier')

parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')

parser.add_argument('--modelpath', type=str, default='./models/',
                    help='path to save the final model')

parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--traindir', type=str, default='./data/train.txt',
                    help='directory with training spectrograms')

parser.add_argument('--testdir', type=str, default='./data/test.txt',
                    help='directory with test spectrograms')

parser.add_argument('--testmodel', type=str, default='NA',
                    help='name of pre-trained model you want to test')

parser.add_argument('--adaptor', type=str, default=None,
                    help='file path of adaptor that will get subtracted from test')

parser.add_argument('--nmels', type=int, default=20,
                    help='number of mel features')

parser.add_argument('--bsz', type=int, default=10,
                    help='batch_size')

parser.add_argument('--split_prop', type=float, default=0.8,
                    help='proportion of training and validation split')

parser.add_argument('--split_method', type=int, default=1,
                    help='split by person: 1, split by tokens: 0')

parser.add_argument('--epoch_prop', type=float, default=1,
                    help='Value between 0 and 1. Determins what percent of training data network sees per epoch')

parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')

parser.add_argument('--hidden_size', type=int, default=10,
                    help='number of hidden units')

args = parser.parse_args()

# DEFINE PARAMETERS
# Input features
input_size = args.nmels  #number of features per time step 
num_classes = 2

#Hyper-parameters
num_epochs = args.epochs
num_layers = args.nlayers
batch_size = args.bsz
hidden_size = args.hidden_size
learning_rate = args.lr

#sequence_length = 50
#sequence_length = 10
sequence_length = 107

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



# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SETUP
# Retrieve training and test data
dat = data.Melspectrogram(args.traindir, args.testdir, args.split_prop, args.split_method, args.adaptor)

# print(len(dat.train))
# print(type(dat.train[0]))

train_data = get_batch(dat.train, batch_size)
train_labs = get_batch(dat.train_labs, batch_size)
train_seq_lens = get_batch(dat.train_seq_lens, batch_size)


#order batches according to sequence length
for i in range(len(train_seq_lens)):
    zipped = zip(list(train_seq_lens[i]), train_data[i], train_labs[i])
    ordered = reversed(sorted(zipped,  key = lambda x: x[0]))
    train_seq_lens[i], train_data[i], train_labs[i] = zip(*ordered)


val_data = dat.val
val_labs = dat.val_labs
val_seq_lens = dat.val_seq_lens

test_data = dat.test
test_labs = dat.test_labs
test_filenames = dat.test_files
test_seq_lens = dat.test_seq_lens

#print(train_data)
#print(type(dat.train))
# Initialize model
model = model.BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# TRAINING and VALIDATION

if args.testmodel == 'NA': #i.e. there isn't a pre-trained model
    ## Train
    total_step = len(train_data)
    for epoch in range(num_epochs):
        subset = get_random_sample(train_data, args.epoch_prop)
        for i,batch in enumerate(train_data): 
            grouped = torch.stack(batch)  #combines list of tensors into a dimension

            sound = grouped.reshape(batch_size, sequence_length, input_size).to(device)
            label = torch.LongTensor(train_labs[i]).to(device)  #labels need to be a tensor of tensors
            seq_len = torch.LongTensor(train_seq_lens[i]).to(device)

            #forward pass
            output = model(sound, seq_len)  
            loss = criterion(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))

    modelname = args.modelpath + 'model' + str(input_size) + '-' + str(learning_rate) + '-' + str(num_epochs) + '-' +  str(num_layers) + '-' + str(hidden_size) + '.pt'

    with open(modelname, 'wb') as f:
                torch.save(model, f)

    ##  Validate (i.e. test on held out natural speech)
    with torch.no_grad():
        correct = 0
        total = 0

        for i, sound in enumerate(val_data):
            sound = sound.reshape(1, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
            print(val_labs[i])
            label = torch.LongTensor(val_labs[i]).to(device)
            seq_len = torch.LongTensor(val_seq_lens[i]).to(device)
            output = model(sound, seq_len)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        if total != 0:
            print('Test Accuracy of the model on natural speech {} %'.format(100 * correct / total)) 



## TEST

if args.testmodel != 'NA':  #if you want to test a pre-existing model, load the model. 
    with open(args.testmodel, 'rb') as f:
        model = torch.load(f)

with torch.no_grad():
    correct = 0
    total = 0
    filenum_dict = dict((el, []) for el in range(1,21))
    probs_dict = {0: [], 1: []}

    for i, sound in enumerate(test_data):
        sound = sound.reshape(1, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
        label = test_labs[i].to(device)
        seq_len = test_seq_lens[i].to(device)
        output = model(sound, seq_len)
        filenum = int(re.findall('\d+', test_filenames[i])[0])
        
        sm = torch.nn.Softmax()
        probs = sm(output)
        prob, lab = torch.topk(probs, 2)
        if int(lab[0][0]) == 0:
            probs_dict[0].append(round(float(prob[0][0]), 3))
            probs_dict[1].append(round(float(prob[0][1]), 3))
            filenum_dict[filenum].append(round(float(prob[0][0]), 3))  # appends the prob of D
        else:
            probs_dict[1].append(round(float(prob[0][0]), 3))
            probs_dict[0].append(round(float(prob[0][1]), 3))
            filenum_dict[filenum].append(round(float(prob[0][1]), 3)) # appends the prob of D

    for i, item in enumerate(probs_dict[0]):
        print(item, test_filenames[i])
        if (i+1)%20 == 0:
            print('-------------------')
        if (i+1)%80 == 0:
            print('#####################################')








    # for key in filenum_dict:  # Print by filenum
    #     for item in filenum_dict[key]:
    #         print(item)
    #     print('----------')


        #Uncomment in order to get a prediction and accuracy. 
    #     _, predicted = torch.max(output.data, 1)
    #     total += label.size(0)
    #     correct += (predicted == label).sum().item()

    # print('Test Accuracy of the model on synthetic speech {} %'.format(100 * correct / total))
