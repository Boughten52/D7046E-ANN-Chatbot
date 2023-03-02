import torch.nn as nn


class ChatBotANN(nn.Module):
    """Defines a feed-forward neural network with three layers. No softmax since it's built-in with cross entropy"""

    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotANN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
