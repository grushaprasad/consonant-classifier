import torch 
import torch.nn as nn
import argparse

import data
import model


# Hyper-parameters
sequence_length = 23
input_size = 128  #number of features per time step 
hidden_size = 128  # can be anything
num_classes = 2
num_epochs = 2
learning_rate = 0.003
num_layers = 2
num_classes = 2

# Device configuration  -- WHAT DOES THIS DO?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch consonant classifier')

parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')

parser.add_argument('--modelname', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--datadir', type=str, default='./data',
                    help='data directory')

# parser.add_argument('--batchsize', type=int, default=1,
#                     help='upper epoch limit') #setting this to 1 will mean that it is not run in batches


args = parser.parse_args()



dat = data.Spectrogram(args.datadir)

train_data = dat.train
test_data = dat.test


model = model.BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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


# Test the model
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






















