import torch

class FFN(torch.nn.Module):
    def __init__(self, observation_size, n_actions, hidden_size=200):
        super(FFN, self).__init__()
        self.observation_size = observation_size
        self.n_actions  = n_actions
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.observation_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.n_actions)

        self.__init_weights(self.fc1)
        self.__init_weights(self.fc2)


    def __init_weights(self, layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.weight.data.fill_(0.)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x