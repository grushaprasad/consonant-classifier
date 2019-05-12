import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_cons, num_vowels, classification):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classification = classification
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.combined = nn.Linear(hidden_size*2, num_vowels*num_cons)  # 2 for bidirection
        self.vowel = nn.Linear(hidden_size*2, num_vowels)
        self.cons = nn.Linear(hidden_size*2, num_cons)
    
    def forward(self, x, seq_length):
        # Pack input
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths = seq_length)
        
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        packed_out, _ = self.lstm(packed, (h0, c0))   #This packed_out has information about the hidden state for every time step for every sequence in the batch
        
        #print(hn.shape)
        # return(x[0])  #returns the last hidden vector?

        # Unpack output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(seq_length) - 1).view(-1, 1).expand(
            len(seq_length), out.size(2))
        time_dimension = 1  #should be 0 if not batch_first
        idx = idx.unsqueeze(time_dimension)
        if out.is_cuda:
            idx = idx.cuda(out.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = out.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        # # Decode the hidden state of the last time step  
        if self.classification == 1:
            out = self.combined(last_output)
            return(out)
        else:
            out_v = self.vowel(last_output)
            out_c = self.cons(last_output)   
            return (out_c, out_v)


"""
To do:

Have two linear layers: One for vowel and one for consonant. Return two outputs (out_cons, out_vowel)
Separately compute the loss for both these outputs in main.py

loss = sum of both these losses. Then backprop on this loss. 

"""
