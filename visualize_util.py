from data_util.experiment_data_classes import Parameters
from data_util.experiment_data_classes import DeepQParameters
import torch
import numpy as np
import time


def visualize_best_path(world, params: Parameters, q_table):
    state = world.reset_agent()
    last_state = state
    reward_sum = 0
    done = False
    for step in range(params.max_steps_per_episode):
        action = np.argmax(q_table[state, :])
        new_state, reward = world.agent_perform_action(action)
        if new_state == last_state:
            done = True
        last_state = state
        state = new_state
        reward_sum += reward
        world.redraw_agent()
        if done:
            break
        time.sleep(0.1)


def visualize_best_path_deep_q(world, params: DeepQParameters, target_net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = world.reset_agent()
    last_state = state
    reward_sum = 0
    done = False
    for step in range(params.max_steps_per_episode):
        state_coords = np.asarray(world.agent_pos).astype(np.float32)
        with torch.no_grad():
            action = target_net(torch.tensor(state_coords).to(device)).argmax(dim=0).to(device)
        new_state, reward = world.agent_perform_action(action)
        if new_state == last_state:
            done = True
        last_state = state
        state = new_state
        reward_sum += reward
        world.redraw_agent()
        # if done:
        #     break
        time.sleep(0.1)
