import torch


class ContinuousActionValueNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_dims:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims + action_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], 1))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state, action):
        x = torch.concatenate([state, action], dim=1)
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class DiscretePolicyNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_nums:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_nums))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = torch.nn.functional.softmax(self.layers[-1](x), dim=-1)
        return x
    

class ContinuousPolicyNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, action_dims:int, hidden_size:list, log_std_range=(-20, 2)):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_dims))  # second to last layer (-2) mean of action
        self.layers.append(torch.nn.Linear(hidden_size[-1], action_dims))  # last layer (-1) log_std of action
        self.layers = torch.nn.ModuleList(self.layers)

        self.log_std_range = log_std_range

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-2):
            x = torch.nn.functional.relu(self.layers[i](x))
        mean = self.layers[-2](x)
        log_std = torch.clamp(self.layers[-1](x), *self.log_std_range)
        return mean, log_std
    

class StateValueNetwork(torch.nn.Module):
    def __init__(self, state_dims:int, hidden_size:list):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(state_dims, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(torch.nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(torch.nn.Linear(hidden_size[-1], 1))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, state):
        x = state
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


if __name__ == '__main__':
    q_net = ContinuousActionValueNetwork(state_dims=8, action_dims=3, hidden_size=[64]).to(torch.device('cpu'))
    pi_net = ContinuousPolicyNetwork(state_dims=8, action_dims=3, hidden_size=[64]).to(torch.device('cpu'))
    v_net = StateValueNetwork(state_dims=8, hidden_size=[64]).to(torch.device('cpu'))
    pi_net = DiscretePolicyNetwork(state_dims=8, action_nums=3, hidden_size=[64]).to(torch.device('cpu'))