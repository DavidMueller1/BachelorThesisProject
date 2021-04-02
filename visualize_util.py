from data_util.experiment_data_classes import Parameters
from data_util.experiment_data_classes import DeepQParameters
from logger import Logger
import torch as T
import numpy as np
import time
from visualization_engine_3d.engine import Engine3D
from tkinter import *


class DeepQInfos:

    def __init__(self, master: Tk, world: Engine3D, params: DeepQParameters, target_net):
        self.current_step = 0
        self.params = params
        self.target_net = target_net

        self.master = master
        self.world = world
        master.title("Infos")

        self.master.bind('r', self.reset)
        self.master.bind('.', self.next)
        self.master.bind(',', self.previous)

    def reset(self, event=None):
        Logger.status("Resetting...")
        self.current_step = 0
        self.world.reset_agent()
        self.world.redraw_agent()

    def next(self, event):
        Logger.debug("next")
        observation = self.world.get_state_for_deep_q(step=self.current_step, max_steps=self.params.max_steps_per_episode)
        # observation = world.get_state_for_deep_q()
        state = T.tensor([observation]).to(self.target_net.device)
        actions = self.target_net.forward(state)
        action = T.argmax(actions).item()
        new_state, reward, done = self.world.agent_perform_action(action)
        # if new_state == last_state:
        #     done = True
        last_state = state
        state = new_state
        # reward_sum += reward
        self.world.redraw_agent()

    def previous(self, event):
        Logger.debug("previous")


def visualize_best_path(world, params: Parameters, q_table):
    state = world.reset_agent()
    last_state = state
    reward_sum = 0
    done = False
    path = [state]
    for step in range(params.max_steps_per_episode):
        action = np.argmax(q_table[state, :])
        new_state, reward, _ = world.agent_perform_action(action)
        if new_state == last_state:
            done = True
            break
        last_state = state
        state = new_state
        path.append(state)
        reward_sum += reward
        world.plot_path(path)
        if done:
            break
        world.redraw_agent()
        time.sleep(0.1)


def visualize_best_path_deep_q(world, params: DeepQParameters, target_net):
    state = world.reset_agent()
    last_state = state
    reward_sum = 0
    path = [state]

    repeat_count = 0
    repeat_max = 6
    for step in range(params.max_steps_per_episode):
        state_coords = np.asarray(world.agent_pos).astype(np.float32)
        adjacent_heights = np.asarray(world.get_agent_adjacent_heights()).astype(np.float32)
        combined_state = np.asarray(world.agent_pos + world.get_agent_adjacent_heights()).astype(np.float32)
        # with T.no_grad():
            # action = target_net(torch.tensor(state_coords).to(device)).argmax(dim=0).to(device)
        observation = world.get_state_for_deep_q(step=step, max_steps=params.max_steps_per_episode)
        # observation = world.get_state_for_deep_q()
        tensor_state = T.tensor([observation]).to(target_net.device)
        actions = target_net.forward(tensor_state)
        action = T.argmax(actions).item()
            # action = target_net(T.tensor(combined_state).to(device)).argmax(dim=0).to(device)
        new_state, reward, done = world.agent_perform_action(action)
        # if new_state == last_state:
        #     repeat_count += 1
        #     if repeat_count > repeat_max:
        #         break
        # else:
        #     repeat_count = 0
        last_state = state
        state = new_state
        path.append(state)
        reward_sum += reward
        world.plot_path(path)
        if done:
            break
        world.redraw_agent()
        time.sleep(0.1)


def get_reward_from_trained_net_on_environment(world, params: DeepQParameters, target_net):
    world.reset_agent()
    state = world.get_state_for_deep_q()
    last_state = state
    reward_sum = 0
    for step in range(params.max_steps_per_episode):
        state_coords = np.asarray(world.agent_pos).astype(np.float32)
        adjacent_heights = np.asarray(world.get_agent_adjacent_heights()).astype(np.float32)
        combined_state = np.asarray(world.agent_pos + world.get_agent_adjacent_heights()).astype(np.float32)
        observation = world.get_state_for_deep_q(step=step, max_steps=params.max_steps_per_episode)
        state = T.tensor([observation]).to(target_net.device)
        actions = target_net.forward(state)
        action = T.argmax(actions).item()
        new_state, reward, done = world.agent_perform_action(action)
        last_state = state
        state = new_state
        reward_sum += reward
        if done:
            break

    return reward_sum
