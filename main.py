import torch 
import torch.nn as nn
import argparse

import data
import model

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='PyTorch consonant classifier')

parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')

parser.add_argument('--modelname', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--traindir', type=str, default='./data/train/',
                    help='directory with training .wav files')

parser.add_argument('--testdir', type=str, default='./data/test/',
                    help='directory with test .wav files')

parser.add_argument('--testmodel', type=str, default='NA',
                    help='name of pre-trained model you want to test')

args = parser.parse_args()

# DEFINE PARAMETERS
# Input features
sequence_length = 23
input_size = 128  #number of features per time step 
num_classes = 2

#Hyper-parameters
num_epochs = args.epochs
num_layers = args.nlayers
hidden_size = 10
learning_rate = 0.003

# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# SETUP
# Retrieve training and test data
dat = data.Melspectrogram(args.traindir, args.testdir)
train_data = dat.train
test_data = dat.test

# Initialize model
model = model.BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# TRAINING

if args.testmodel == 'NA': #i.e. there isn't a pre-trained model
    total_step = len(train_data)
    for epoch in range(num_epochs):
        for i, (sound, label) in enumerate(train_data):  
            sound = sound.reshape(1, sequence_length, input_size).to(device) #change this 1 to batch size if I want to implement batches in the future
            label = label.to(device)  #gets the index of the label and writes to to device

            #forward pass
            output = model(sound)  
            loss = criterion(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))

    with open(args.modelname, 'wb') as f:
                torch.save(model, f)


# TEST

if args.testmodel != 'NA':  #if you want to test a pre-existing model, load the model. 
    with open(args.testmodel, 'rb') as f:
        model = torch.load(f)

with torch.no_grad():
    correct = 0
    total = 0
    for sound, label in test_data:
        sound = sound.reshape(1, sequence_length, input_size).to(device)
        label = label.to(device)
        output = model(sound)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('Test Accuracy of the model {} %'.format(100 * correct / total))


