import time
from plot_util import plot_progress
from util.time_estimate import TimeEstimater
from plot_util import plot_network
from plot_util import plot_network_layer
from logger import Logger

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_states, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=5)
        # self.fc3 = nn.Linear(in_features=5, out_features=5)
        self.out = nn.Linear(in_features=5, out_features=num_actions)
        # self.fc1 = nn.Linear(in_features=num_states, out_features=5)
        # self.fc2 = nn.Linear(in_features=5, out_features=5)
        # self.fc3 = nn.Linear(in_features=5, out_features=4)
        # self.out = nn.Linear(in_features=4, out_features=num_actions)

    def forward(self, t):
        # t = t.flatten(start_dim=0)
        # t = t.reshape([-1])
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        # t = F.relu(self.fc3(t))
        # t = self.fc1(t)
        # t = self.fc2(t)
        # t = self.fc3(t)
        t = self.out(t)
        return t


Sars = namedtuple(
    'Sars',
    ('state', 'action', 'reward', 'next_state')
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.push_count = 0

    def push(self, sars):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sars)
        else:
            self.buffer[self.push_count] = sars
        self.push_count = (self.push_count + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_one(self):
        return random.choice(self.buffer)

    def can_provide_sample(self, batch_size):
        return len(self.buffer) >= batch_size


class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # Logger.debug(policy_net(torch.tensor([5.]).to(QValues.device)))
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        # final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        # non_final_state_locations = (final_state_locations is False)
        # non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        # values[next_states] = target_net(next_states).max(dim=1)[0].detach()
        values = target_net(next_states).max(dim=1)[0].detach()
        return values


def extract_tensors(sars):
    batch = Sars(*zip(*sars))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


def train(width: int, length: int, params, environment, visualize=False, plot=False, plot_interval=10, plot_moving_avg_period=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = ReplayBuffer(params.replay_buffer_size)
    # buffer = []
    layer_1_values = []
    layer_2_values = []
    layer_out_values = []

    policy_net = DQN(7, 4).float().to(device)
    policy_net.train()
    target_net = DQN(7, 4).float().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=params.learning_rate)

    time_estimater = TimeEstimater(params.num_episodes)

    exploration_rate = params.start_exploration_rate
    for episode in range(params.num_episodes):
        if episode % 1000 == 0:
            Logger.debug("EPISODE: ", episode)
        state = environment.reset_agent()
        done = False
        rewards_current_episode = 0
        max_reward_current_episode = 0
        for step in range(params.max_steps_per_episode):
            exploration_rate_threshold = random.uniform(0, 1)
            state_coords = np.asarray(environment.agent_pos).astype(np.float32)
            adjacent_heights = np.asarray(environment.get_agent_adjacent_heights()).astype(np.float32)
            combined_state = np.asarray(environment.agent_pos + environment.get_agent_adjacent_heights()).astype(np.float32)
            if exploration_rate_threshold > exploration_rate:
                with torch.no_grad():
                    # Logger.debug("Values:", policy_net(torch.tensor([float(state)]).to(device)))
                    # action = policy_net(torch.tensor([float(state)]).to(device)).argmax(dim=0).to(device)
                    action = policy_net(torch.tensor(combined_state).to(device)).argmax(dim=0).to(device)
                    # action = policy_net(torch.tensor(state_coords).to(device)).argmax(dim=0).to(device)
            else:
                action = random.choice(environment.get_agent_possible_actions())
                action = torch.tensor([action]).to(device)
            new_state, reward = environment.agent_perform_action(action.item())

            new_state_coords = np.asarray(environment.agent_pos).astype(np.float32)
            new_adjacent_heights = np.asarray(environment.get_agent_adjacent_heights()).astype(np.float32)
            new_combined_state = np.asarray(environment.agent_pos + environment.get_agent_adjacent_heights()).astype(np.float32)

            rewards_current_episode += reward
            replay_buffer.push(Sars(torch.tensor(combined_state).to(device), action, torch.tensor([float(reward)]).to(device), torch.tensor(new_combined_state).to(device)))
            # buffer.append(Sars(torch.tensor([float(state)]).to(device), action, torch.tensor([float(reward)]).to(device), torch.tensor([float(new_state)]).to(device)))
            # buffer.append(Sars(torch.tensor(combined_state).to(device), action, torch.tensor([float(reward)]).to(device), torch.tensor(new_combined_state).to(device)))
            # buffer.append(Sars(torch.tensor(state_coords).to(device), action, torch.tensor([float(reward)]).to(device), torch.tensor(new_state_coords).to(device)))
            state = new_state
            if max_reward_current_episode < reward:
                max_reward_current_episode = reward

            sars = replay_buffer.sample_one()
            state, action, reward, next_state = sars
            # state, action, reward, next_state = extract_tensors(sars)

            current_q_values = policy_net(state)
            # current_q_values = QValues.get_current(policy_net, state, action)
            # next_q_values = QValues.get_next(target_net, next_state)
            next_q_values = target_net(next_state)
            # target_q_values = reward + (next_q_values * params.discount_rate)
            target_q_values = current_q_values.clone()
            target_q_values[action.item()] = reward + torch.max(next_q_values) * params.discount_rate
            # target_q_values[action.item()] = target_q_values[action.item()] * (1 - params.learning_rate) + params.learning_rate * (reward + torch.max(next_q_values) * params.discount_rate)

            # Logger.debug("Action:", action)
            # Logger.debug("Reward:", reward)
            # Logger.debug("Current:", current_q_values)
            # Logger.debug("Next:", next_q_values)
            # Logger.debug("Max:", torch.max(next_q_values))
            # Logger.debug("Target:", target_q_values)
            # Logger.debug("----------")
            # Logger.debug("Reward:", reward)
            # Logger.debug("Target Q-Values:", target_q_values)

            loss = F.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if visualize:
                environment.redraw_agent()
                time.sleep(0.04)


        # experiences = buffer.sample_one()
        # random.shuffle(buffer)
        # while len(buffer) > 0:
        #     state, action, reward, next_state = buffer.pop(0)
        #     # states, actions, rewards, next_states = extract_tensors(experiences)
        #     # state, action, reward, next_state = extract_tensors(experiences)
        #
        #     # current_q_values = QValues.get_current(policy_net, states, actions)
        #     current_q_values = policy_net(state)
        #     # next_q_values = QValues.get_next(target_net, next_state)
        #     next_q_values = target_net(next_state)
        #     target_q_values = current_q_values.clone()
        #     target_q_values[action.item()] = reward + torch.max(next_q_values) * params.discount_rate
        #     # Logger.debug("Action:", action)
        #     # Logger.debug("Reward:", reward)
        #     # Logger.debug("Current:", current_q_values)
        #     # Logger.debug("Next:", next_q_values)
        #     # Logger.debug("Max:", torch.max(next_q_values))
        #     # Logger.debug("Target:", target_q_values)
        #     # Logger.debug("----------")
        #     # target_q_values = (next_q_values * params.discount_rate) * reward
        #     # Logger.debug("Reward:", reward)
        #     # Logger.debug("Target Q-Values:", target_q_values)
        #     # target_q_values = current_q_values * reward
        #
        #     loss = F.mse_loss(current_q_values, target_q_values)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        if episode % params.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            layer_1_values.append(target_net.fc1.weight.cpu().data.numpy())
            layer_2_values.append(target_net.fc2.weight.cpu().data.numpy())
            layer_out_values.append(target_net.out.weight.cpu().data.numpy())
            # plot_network(target_net)
            plot_network_layer(4, "Layer 1", layer_1_values, episode)
            plot_network_layer(5, "Layer 2", layer_2_values, episode)
            plot_network_layer(6, "Layer Out", layer_out_values, episode)

        exploration_rate = params.min_exploration_rate + (params.max_exploration_rate - params.min_exploration_rate) * np.exp(
            -params.exploration_decay_rate * episode)

        params.rewards_all_episodes.append(rewards_current_episode)
        params.max_rewards_all_episodes.append(max_reward_current_episode)

        if episode % plot_interval == 0 and plot:
            plot_progress(params.rewards_all_episodes, exploration_rate, plot_moving_avg_period, time_left=time_estimater.get_time_left(episode))

    return target_net, params
