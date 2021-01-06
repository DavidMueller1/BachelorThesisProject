from data_util.experiment_data_classes import Parameters
from data_util.experiment_data_classes import DeepQParameters
from logger import Logger
import torch as T
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
    # device = T.device("cuda" if T.cuda.is_available() else "cpu")

    world.reset_agent()
    state = world.get_state_for_deep_q()
    last_state = state
    reward_sum = 0
    for step in range(params.max_steps_per_episode):
        state_coords = np.asarray(world.agent_pos).astype(np.float32)
        adjacent_heights = np.asarray(world.get_agent_adjacent_heights()).astype(np.float32)
        combined_state = np.asarray(world.agent_pos + world.get_agent_adjacent_heights()).astype(np.float32)
        # with T.no_grad():
            # action = target_net(torch.tensor(state_coords).to(device)).argmax(dim=0).to(device)
        observation = np.append(world.get_state_for_deep_q(), np.array(params.max_steps_per_episode, dtype=np.float32))
        # observation = world.get_state_for_deep_q()
        state = T.tensor([observation]).to(target_net.device)
        actions = target_net.forward(state)
        action = T.argmax(actions).item()
            # action = target_net(T.tensor(combined_state).to(device)).argmax(dim=0).to(device)
        new_state, reward, done = world.agent_perform_action(action)
        # if new_state == last_state:
        #     done = True
        last_state = state
        state = new_state
        reward_sum += reward
        world.redraw_agent()
        if done:
            break
        time.sleep(0.1)
