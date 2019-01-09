import torch 
import torch.nn as nn
import argparse

import data
import model

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='PyTorch consonant classifier')

parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')

parser.add_argument('--modelpath', type=str, default='./',
                    help='path to save the final model')

parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--traindir', type=str, default='./data/64mels/train/',
                    help='directory with training spectrograms')

parser.add_argument('--testdir', type=str, default='./data/64mels/test/natural/',
                    help='directory with test spectrograms')

parser.add_argument('--testmodel', type=str, default='NA',
                    help='name of pre-trained model you want to test')

parser.add_argument('--nmels', type=int, default=64,
                    help='number of mel features')

parser.add_argument('--bsz', type=int, default=10,
                    help='batch_size')

args = parser.parse_args()

# DEFINE PARAMETERS
# Input features
input_size = args.nmels  #number of features per time step 
num_classes = 2

#Hyper-parameters
num_epochs = args.epochs
num_layers = args.nlayers
batch_size = args.bsz
hidden_size = 10
learning_rate = 0.003


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def split(l, n):
    # modified from: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115
    k,m = divmod(len(l), round(len(l)/n))
    new = [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    return(new)
    #return(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# SETUP
# Retrieve training and test data
dat = data.Melspectrogram(args.traindir, args.testdir)

train_data = split(dat.train, batch_size)
train_labs = split(dat.train_labs, batch_size)
test_data = dat.test
test_labs = dat.test_labs

sequence_length = dat.max_seq_length

print('Max sequence length: {}'.format(sequence_length))

# Initialize model
model = model.BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # TRAINING

if args.testmodel == 'NA': #i.e. there isn't a pre-trained model
    total_step = len(train_data)
    for epoch in range(num_epochs):
        for i,batch in enumerate(train_data):
            grouped = torch.stack(batch)  #combines list of tensors into a dimension
            sound = grouped.reshape(batch_size, sequence_length, input_size).to(device)
            label = torch.LongTensor(train_labs[i]).to(device)  #labels need to be a tensor of tensors

            #forward pass
            output = model(sound)  
            loss = criterion(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))

    modelname = args.modelpath + 'model' + str(input_size) + '-' + str(num_epochs) + '-' + str(num_layers) + '-' + str(hidden_size) + '.pt'

    with open(modelname, 'wb') as f:
                torch.save(model, f)

## TEST

if args.testmodel != 'NA':  #if you want to test a pre-existing model, load the model. 
    with open(args.testmodel, 'rb') as f:
        model = torch.load(f)

with torch.no_grad():
    correct = 0
    total = 0

    for i, sound in enumerate(test_data):
        sound = sound.reshape(1, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
        label = test_labs[i].to(device)
        output = model(sound)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('Test Accuracy of the model {} %'.format(100 * correct / total))
