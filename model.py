import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x, seq_length):
        # Pack output
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths = seq_length)
        
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        packed_out, _ = self.lstm(packed, (h0, c0))

        # Unpack output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)

        #print(out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])   # is this doing the right thing?? Should I get non

        return out
