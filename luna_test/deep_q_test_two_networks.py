import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims, n_actions, batch_size, target_update, max_mem_size=100000, eps_end=0.01, eps_dec=0.0001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eps_min = eps_end
        self.eps_max = epsilon
        self.eps_dec = eps_dec
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory_counter = 0

        self.policy_net = DeepQNetwork(self.learning_rate, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.policy_net.train()
        self.target_net = DeepQNetwork(self.learning_rate, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.target_net.eval()
        self.best_net = DeepQNetwork(self.learning_rate, n_actions=n_actions, input_dims=input_dims, fc1_dims=256,
                                       fc2_dims=256)
        self.best_net.eval()

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def calculate_epsilon(self, episode):
        self.epsilon = self.eps_min if self.eps_min >= self.epsilon else self.eps_min + (
                self.eps_max - self.eps_min) * np.exp(
            -self.eps_dec * episode)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.policy_net.device)
            actions = self.policy_net.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, episode):
        if self.memory_counter < self.batch_size:
            return

        self.policy_net.optimizer.zero_grad()

        max_mem = min(self.memory_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.policy_net.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.policy_net.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.policy_net.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.policy_net.device)

        action_batch = self.action_memory[batch]

        q_eval = self.policy_net.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_net.forward(new_state_batch)  # Target network would be used here
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.policy_net.loss(q_target, q_eval).to(self.policy_net.device)
        loss.backward()
        self.policy_net.optimizer.step()

        if episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # layer_1_values.append(target_net.fc1.weight.cpu().data.numpy())
            # layer_2_values.append(target_net.fc2.weight.cpu().data.numpy())
            # layer_out_values.append(target_net.out.weight.cpu().data.numpy())

        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
