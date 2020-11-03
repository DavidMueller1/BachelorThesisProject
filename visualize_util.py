from data_util.experiment_data_classes import Parameters
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