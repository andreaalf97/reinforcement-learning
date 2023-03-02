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

        # TODO: Fix the initialization. If I add this now the network will always generate
        # the same output with any input tensor.
        # self.__init_weights(self.fc1)
        # self.__init_weights(self.fc2)


    def __init_weights(self, layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.weight.data.fill_(0.)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class ConvBrain(torch.nn.Module):
    def __init__(self, observation_size: tuple, n_actions: int, in_channels=1, hidden_channels=32, kernel_size=4, padding=2, hidden_size=100):
        super(ConvBrain, self).__init__()

        assert isinstance(observation_size, tuple) and len(observation_size) == 2, f"Expected tuple of size 2 as `observation_size`, received {type(observation_size)}"
        assert isinstance(n_actions, int), f"Expected int as `n_actions`, received {type(n_actions)}"

        self.observation_size = observation_size
        self.n_actions = n_actions

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.hidden_size = hidden_size

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.hidden_channels,
            self.kernel_size,
            padding=self.padding,
        )

        self.fc1 = torch.nn.Linear((self.observation_size[0]+1) * (self.observation_size[1]+1) * (self.hidden_channels), self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.n_actions)

        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x