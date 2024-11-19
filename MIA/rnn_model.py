'''
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # LSTM output shape: (batch_size, seq_length, hidden_size)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        # h_n contains the hidden state for t = seq_length
        out, _ = self.lstm(x)
        # We use the last hidden state to classify
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)
        return out

# Example usage:
# Assuming input_size is the number of features in your input (e.g., 16 for ConvNet's fc1)
# hidden_size can be any number you choose (e.g., 128)
# output_size is the number of classes (e.g., 10 for classification)
input_size = 16  # This should match the input size of your data
hidden_size = 128  # You can choose this size
output_size = 10  # This should match the number of classes you have

# Create the RNN model
rnn_model = SimpleRNN(input_size, hidden_size, output_size)

# Example input tensor (batch_size=4, seq_length=5, input_size=16)
example_input = torch.randn(4, 5, input_size)

# Forward pass through the RNN
output = rnn_model(example_input)
print(output)  # This will print the output of the RNN which can be used for classification

'''

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, height, width, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        input_size = height * width  # calculate input_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # flatten to (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)
        return out

# Example usage:
height = 28
width = 28
hidden_size = 128
output_size = 10

rnn_model = SimpleRNN(height, width, hidden_size, output_size)

example_input = torch.randn(4, 5, 28, 28)  # Assuming input shape is (batch_size, seq_length, height, width)

output = rnn_model(example_input)
#print(output)
